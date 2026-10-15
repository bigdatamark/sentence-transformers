#!/usr/bin/env python3
# [MODULE DOC]
# Example script: embedding-level knowledge distillation. A student SentenceTransformer
# (DistilBERT) is trained with MSELoss to reproduce a teacher's (all-mpnet-base-v2) embeddings of
# unique AllNLI sentences; quality is tracked with an STS-B validation EmbeddingSimilarityEvaluator.
# The module docstring documents alternative distillation patterns and dimension-mismatch handling.
# Module-level constants name the base model(s), datasets, sizes, output directory and run name;
# SMOKE_TEST is true when the SMOKE_TEST environment variable equals "1" and shrinks data to a few
# rows, limits training to one step and skips the Hub push. Running the file as a script calls main.
# The inline script metadata block lists dependencies for "uv run".
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers[train]>=5.0",
#     "datasets>=2.19.0",
#     "accelerate>=0.26.0",
#     "trackio",
# ]
# ///
"""Distillation template: train a small student to match a stronger teacher's embeddings.

This script implements **embedding MSE** (Pattern 1 below): pre-compute teacher
embeddings, train the student to minimize MSE against them. Yields a smaller /
faster student typically reaching 95-99% of teacher quality.

Three distillation patterns:
- Pattern 1 (this script): `(text, teacher_embedding)` + `MSELoss`. Cheapest,
  most data-efficient. Use when student + teacher are both bi-encoders with the
  same output dim and you have a pile of unlabeled text.
- Pattern 2: `(query, positive, negative, score_diff)` + `MarginMSELoss` from a
  CrossEncoder teacher's score differences. Workhorse of ms-marco distillation.
  See `../references/losses_sentence_transformer.md` (MarginMSELoss section).
- Pattern 3: `(query, positive, neg_1, ..., neg_n, labels)` + `DistillKLDivLoss`
  to preserve the full teacher distribution. More data-hungry. Natural fit when
  distilling from an ensemble of rerankers.

Mismatched dims: if the student's dim is smaller than the teacher's, MSELoss
fails. Add a PCA-init `Dense` projection so the teacher matches the student:

    from sklearn.decomposition import PCA
    from sentence_transformers.sentence_transformer.modules import Dense
    pca = PCA(n_components=student.get_embedding_dimension())
    pca.fit(teacher.encode(sentences[:20_000], convert_to_numpy=True))
    dense = Dense(
        in_features=teacher.get_embedding_dimension(),
        out_features=student.get_embedding_dimension(),
        bias=False, activation_function=torch.nn.Identity(),
    )
    dense.linear.weight = torch.nn.Parameter(torch.from_numpy(pca.components_).float())
    teacher.add_module("dense", dense)

Distilling to a CrossEncoder student: construct with `activation_fn=nn.Identity()`
or eval ranking collapses silently. Every non-BCE CE loss expects raw logits
during training, but the model's `activation_fn` runs at eval time inside
`predict()`. Default `Sigmoid` (when `num_labels=1`) saturates raw logits >5 to
~1.0, dropping nDCG from e.g. ~0.59 to ~0.14 with healthy-looking training loss.
Applies to all CE distillation / listwise / pairwise losses. See SKILL.md
Directive 7 ([CE]).

Layer pruning shortcut for Pattern 1: copy the teacher, delete layers (often
keeps 99%+ of quality at a fraction of the layers), then distill with MSELoss:

    from copy import deepcopy
    student = deepcopy(teacher)
    layers = student.transformers_model.encoder.layer  # BERT/MPNet/DistilBERT
    student.transformers_model.encoder.layer = torch.nn.ModuleList(
        [layers[0], layers[3], layers[6], layers[9]]
    )
    student.transformers_model.config.num_hidden_layers = 4

Tips: pre-compute teacher outputs once and cache (`dataset.save_to_disk`). LR
1e-4 (higher than the usual 2e-5: the target is dense regression). 1 epoch is
usually enough. The student inherits the teacher's weaknesses, so pick a
teacher strong on YOUR task. If the teacher expects an instruction prefix,
include it during teacher encoding so the student's target matches inference.

For multilingual student distillation (extend an English teacher to other
languages without in-language supervised data), see `train_sentence_transformer_make_multilingual_example.py`.
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext

import torch
from datasets import Dataset, load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.losses import MSELoss
from sentence_transformers.sentence_transformer.modules import Normalize
from sentence_transformers.util.similarity import SimilarityFunction


def autocast_ctx():
    # [FUNCTION DOC]
    # HELPER: Returns a context manager for mixed precision around evaluator calls made outside the
    # trainer: a no-op context when CUDA is unavailable, otherwise CUDA autocast in bfloat16 if the GPU
    # supports it, else float16.
    """bf16/fp16 autocast for evaluator calls outside the trainer (which has its own autocast)."""
    if not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def log_trackio_dashboard():
    # [FUNCTION DOC]
    # HELPER: Best-effort log of the Trackio dashboard URL: asks the Hugging Face Hub (network) for the
    # logged-in user name and, if present, logs the URL of that user's "trackio" Space. Every exception
    # (not logged in, offline) is silently swallowed.
    """Surface the Trackio dashboard URL so the user can watch training live."""
    try:
        from huggingface_hub import whoami

        hf_user = whoami().get("name")
        if hf_user:
            logging.info(
                f"Trackio dashboard (live training progress): https://huggingface.co/spaces/{hf_user}/trackio"
            )
    except Exception:
        pass


TEACHER_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
STUDENT_MODEL_NAME = "distilbert/distilbert-base-uncased"

CORPUS_DATASET = "sentence-transformers/all-nli"
CORPUS_SUBSET = "pair"

TRAIN_SIZE = 50_000
EVAL_SIZE = 1_000
OUTPUT_DIR = "models/distilbert-distilled-from-mpnet"
RUN_NAME = "distilbert-distill-from-mpnet"

TEACHER_ENCODE_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 128
SMOKE_TEST = os.environ.get("SMOKE_TEST") == "1"


def setup_logging():
    # [FUNCTION DOC]
    # HELPER: Creates the "logs" directory, force-reconfigures root logging at INFO with a timestamped
    # format writing to both the console and logs/<RUN_NAME>.log, lowers noisy HTTP/hub/filesystem
    # library loggers to WARNING, and on CUDA sets float32 matmul precision to "high" (TF32).
    """Configure logging + TF32. Tees to logs/{RUN_NAME}.log and silences HTTP spam."""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/{RUN_NAME}.log")],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere+, no quality loss


def main() -> None:
    # [FUNCTION DOC]
    # Entry point. Parses only an optional --eval-only model path (unknown arguments, e.g. from
    # launchers, are ignored) and sets up logging. In eval-only mode it loads that saved model, runs
    # an STS-B validation cosine-similarity evaluator under autocast and returns.
    # Otherwise: loads the teacher and the student (with model card). Appends a
    # Normalize module to the student if the teacher has one and the student does not. Exits with
    # SystemExit if the embedding dimensions differ. Collects unique anchor/positive strings from the
    # AllNLI pair subset in order, takes train+eval sentences (50+20 in smoke tests), encodes both
    # parts with the teacher (batch 256) and builds (sentence, label embedding) datasets. Logs the
    # teacher's STS score, then the student baseline. Training: batch size 128, learning rate 1e-4,
    # weight decay 0.01; evaluation here is not wrapped in autocast.
    # Common tail: runs a baseline evaluation first (this also gives the evaluator its name-prefixed
    # primary metric, used as metric_for_best_model with "eval_" prepended), builds training arguments
    # (max 1 step under SMOKE_TEST, linear-style warmup of 10 percent of steps, bf16, evaluation and
    # checkpointing every 10 percent keeping 2 checkpoints, logging every 1 percent, best model
    # reloaded at the end, Trackio reporting except in smoke tests, seed 12), logs the Trackio URL
    # (not in smoke tests), trains, re-evaluates and logs a verdict: WIN if the primary score rose by at
    # least 0.005, MARGINAL if it did not drop, else REGRESSION. Saves the model under the output
    # directory's "final" subfolder and, unless SMOKE_TEST, pushes it to the Hub as RUN_NAME, logging
    # the traceback instead of raising if the push fails.
    # Side effects: dataset/model downloads, GPU compute, writes under models/, logs/ and data/, Hub upload.
    # KNOWN ISSUE: bf16 mixed precision is hard-coded on in the training arguments, so training fails on CPU-only machines or GPUs without bfloat16 support, even though the autocast helper falls back to float16.
    # KNOWN ISSUE: the dimension-mismatch error points to a 'MISMATCHED EMBEDDING DIMS' section, but the module docstring's section is titled 'Mismatched dims'.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only", type=str, default=None, help="Skip training; load this saved model and run only the evaluator."
    )
    cli, _ = parser.parse_known_args()

    setup_logging()

    if cli.eval_only:
        logging.info(f"Eval-only mode: loading model from {cli.eval_only}")
        model = SentenceTransformer(cli.eval_only)
        stsb = load_dataset("sentence-transformers/stsb", split="validation")
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=stsb["sentence1"],
            sentences2=stsb["sentence2"],
            scores=stsb["score"],
            main_similarity=SimilarityFunction.COSINE,
            name="sts-dev",
        )
        with autocast_ctx():
            evaluator(model)
        return

    logging.info(f"Loading teacher: {TEACHER_MODEL_NAME}")
    teacher = SentenceTransformer(TEACHER_MODEL_NAME)

    logging.info(f"Loading student: {STUDENT_MODEL_NAME}")
    student = SentenceTransformer(
        STUDENT_MODEL_NAME,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{STUDENT_MODEL_NAME.split('/')[-1]} distilled from {TEACHER_MODEL_NAME.split('/')[-1]}",
        ),
    )
    # Match the teacher's final Normalize. MSELoss against unit-norm targets fights student
    # outputs at norm ~5-10 and can silently regress
    if any(isinstance(m, Normalize) for m in teacher) and not any(isinstance(m, Normalize) for m in student):
        student.append(Normalize())

    if student.get_embedding_dimension() != teacher.get_embedding_dimension():
        raise SystemExit(
            f"Student dim ({student.get_embedding_dimension()}) != teacher dim "
            f"({teacher.get_embedding_dimension()}). Plain MSELoss requires matching dims. "
            "Either pick a student with matching dim, or add a Dense projection layer "
            "(PCA-initialized from teacher embeddings). See the 'MISMATCHED EMBEDDING DIMS' "
            "section in this script's docstring."
        )

    logging.info(f"Loading corpus: {CORPUS_DATASET} ({CORPUS_SUBSET})")
    train_size = 50 if SMOKE_TEST else TRAIN_SIZE
    eval_size = 20 if SMOKE_TEST else EVAL_SIZE
    if SMOKE_TEST:
        logging.info("SMOKE_TEST=1: trimmed dataset; will run max_steps=1 and skip Hub push")
    raw = load_dataset(CORPUS_DATASET, CORPUS_SUBSET, split="train")
    sentences = list(dict.fromkeys(s for row in raw for s in (row["anchor"], row["positive"]) if isinstance(s, str)))
    sentences = sentences[: train_size + eval_size]
    train_sentences = sentences[:train_size]
    eval_sentences = sentences[train_size : train_size + eval_size]

    logging.info(f"Encoding {len(train_sentences):,} training sentences with the teacher (may take a while)")
    teacher_train = teacher.encode(
        train_sentences, batch_size=TEACHER_ENCODE_BATCH_SIZE, convert_to_numpy=True, show_progress_bar=True
    )

    logging.info(f"Encoding {len(eval_sentences):,} eval sentences with the teacher")
    teacher_eval = teacher.encode(
        eval_sentences, batch_size=TEACHER_ENCODE_BATCH_SIZE, convert_to_numpy=True, show_progress_bar=True
    )

    train_dataset = Dataset.from_dict({"sentence": train_sentences, "label": teacher_train.tolist()})
    eval_dataset = Dataset.from_dict({"sentence": eval_sentences, "label": teacher_eval.tolist()})

    logging.info(f"Building training dataset ({len(train_dataset):,}) and eval dataset ({len(eval_dataset):,})")

    loss = MSELoss(model=student)

    logging.info("Setting up STS-B evaluator for quality tracking")
    stsb = load_dataset("sentence-transformers/stsb", split="validation")
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=stsb["sentence1"],
        sentences2=stsb["sentence2"],
        scores=stsb["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
    )
    logging.info("Teacher performance:")
    evaluator(teacher)
    logging.info("Student performance before distillation:")
    # Must run before deriving metric_key: evaluator(model) mutates primary_metric to add the name_ prefix.
    baseline_eval = evaluator(student)[evaluator.primary_metric]
    metric_key = f"eval_{evaluator.primary_metric}"

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=1 if SMOKE_TEST else -1,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=0.1,
        bf16=True,
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=2,
        logging_steps=0.01,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model=metric_key,
        greater_is_better=True,
        report_to="none" if SMOKE_TEST else "trackio",
        run_name=RUN_NAME,
        seed=12,
    )

    trainer = SentenceTransformerTrainer(
        model=student,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    if not SMOKE_TEST:
        log_trackio_dashboard()
    trainer.train()

    logging.info("Student performance after distillation:")
    score = evaluator(student)[evaluator.primary_metric]
    delta = score - baseline_eval
    verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
    logging.info(f"VERDICT: {verdict} | score={score:.4f} | baseline={baseline_eval:.4f} | delta={delta:+.4f}")

    final_dir = f"{OUTPUT_DIR}/final"
    student.save_pretrained(final_dir)
    logging.info(f"Saved to {final_dir}")

    if SMOKE_TEST:
        logging.info("SMOKE_TEST=1: skipping Hub push")
        return

    try:
        commit_url = student.push_to_hub(RUN_NAME)
        logging.info(f"Pushed model to {commit_url.rsplit('/commit/', 1)[0]}")
    except Exception:
        import traceback

        logging.error(f"Hub push failed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
