#!/usr/bin/env python3
# [MODULE DOC]
# Example script: multilingual knowledge distillation. A multilingual student
# (XLM-RoBERTa-base) learns, via MSELoss, to map both English sentences and their translations
# (de, es, fr, it from a parallel-sentences dataset) onto an English teacher's embeddings of the
# English side. Evaluated per language with MSE and translation-retrieval evaluators.
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
"""Multilingual teacher-student distillation: extend an English bi-encoder to other languages.

The trick (Reimers & Gurevych 2020, https://huggingface.co/papers/2004.09813):
the teacher embeds English. The multilingual student is trained so that BOTH
the English sentence AND its translation map to the SAME teacher embedding.
Cross-lingual retrieval works out of the box because translations sit near
each other in the joint space.

Use when you have a strong English bi-encoder and want a multilingual version
but lack in-language supervised data. If you DO have in-language labeled data,
train directly with MNRL / CoSENTLoss on it. That usually wins on in-language
tasks.

Data: parallel `(english, non_english)` pairs. `sentence-transformers/parallel-sentences-*`
covers many corpora (talks, europarl, tatoeba, wikimatrix, opensubtitles, jw300,
news-commentary, ...) with `{src}-{tgt}` subsets. ~500k pairs per language is
plenty.

Picks:
- Teacher: any English bi-encoder you want a multilingual copy of
  (all-mpnet-base-v2, all-MiniLM-L6-v2, BAAI/bge-base-en-v1.5, intfloat/e5-base-v2).
- Student: must be multilingual (xlm-roberta-base, paraphrase-multilingual-MiniLM-L12-v2,
  microsoft/mdeberta-v3-base).
- Student dim must match teacher dim, otherwise add a PCA-init Dense projection
  (see train_sentence_transformer_distillation_example.py).
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext

import numpy as np
import torch
from datasets import DatasetDict, load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.evaluation import (
    MSEEvaluator,
    SequentialEvaluator,
    TranslationEvaluator,
)
from sentence_transformers.sentence_transformer.losses import MSELoss
from sentence_transformers.sentence_transformer.modules import Normalize


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
STUDENT_MODEL_NAME = "FacebookAI/xlm-roberta-base"

PARALLEL_DATASET = "sentence-transformers/parallel-sentences-talks"
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGES = ("de", "es", "fr", "it")

MAX_SENTENCES_PER_LANGUAGE = 200_000
EVAL_SENTENCES_PER_LANGUAGE = 1_000
STUDENT_MAX_SEQ_LENGTH = 128

OUTPUT_DIR = "models/xlm-roberta-multilingual-from-mpnet"
RUN_NAME = "xlm-roberta-multilingual-from-mpnet"

TEACHER_ENCODE_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 64
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


def load_parallel_data() -> tuple[DatasetDict, DatasetDict]:
    # [FUNCTION DOC]
    # HELPER: For each target language loads the "<source>-<target>" subset's train split (logging an
    # error and skipping the language if loading fails), capped at MAX_SENTENCES_PER_LANGUAGE rows. The
    # eval set is the first EVAL_SENTENCES_PER_LANGUAGE rows of the dev split; if that fails for any
    # reason, that many rows are split off the train set instead (shuffled, seed 12). Returns
    # (train DatasetDict, eval DatasetDict) keyed by subset name; exits with SystemExit if no language
    # loaded.
    """Load (en, target_lang) parallel pairs for each target language as a DatasetDict."""
    train_dict = DatasetDict()
    eval_dict = DatasetDict()
    for tgt in TARGET_LANGUAGES:
        subset = f"{SOURCE_LANGUAGE}-{tgt}"
        try:
            train_ds = load_dataset(PARALLEL_DATASET, subset, split="train")
        except Exception as exc:
            logging.error(f"Could not load {PARALLEL_DATASET}/{subset}: {exc}")
            continue
        if len(train_ds) > MAX_SENTENCES_PER_LANGUAGE:
            train_ds = train_ds.select(range(MAX_SENTENCES_PER_LANGUAGE))

        try:
            eval_ds = load_dataset(PARALLEL_DATASET, subset, split="dev").select(range(EVAL_SENTENCES_PER_LANGUAGE))
        except Exception:
            split = train_ds.train_test_split(test_size=EVAL_SENTENCES_PER_LANGUAGE, shuffle=True, seed=12)
            train_ds, eval_ds = split["train"], split["test"]

        train_dict[subset] = train_ds
        eval_dict[subset] = eval_ds
    if not train_dict:
        raise SystemExit(f"No language subsets loaded from {PARALLEL_DATASET}. Check TARGET_LANGUAGES.")
    return train_dict, eval_dict


def build_evaluator(eval_dict: DatasetDict, teacher: SentenceTransformer) -> SequentialEvaluator:
    # [FUNCTION DOC]
    # HELPER: For each eval subset adds an MSEEvaluator (student embeddings of English and of the
    # translation versus teacher embeddings of English) and a TranslationEvaluator (cross-lingual
    # matching accuracy), both named by subset. Returns a SequentialEvaluator whose main score is the
    # mean of every second score, i.e. the translation evaluators' scores, assuming the alternating
    # order.
    """Per-language MSE + TranslationEvaluator. `main_score_function` averages
    translation accuracies only. MSE (`negative_mse * 100`) is on a different
    scale and would break the verdict threshold if mixed in."""
    sub_evaluators = []
    for subset, ds in eval_dict.items():
        sub_evaluators.append(
            MSEEvaluator(
                source_sentences=ds["english"],
                target_sentences=ds["non_english"],
                name=subset,
                teacher_model=teacher,
                batch_size=TEACHER_ENCODE_BATCH_SIZE,
            )
        )
        sub_evaluators.append(
            TranslationEvaluator(
                source_sentences=ds["english"],
                target_sentences=ds["non_english"],
                name=subset,
                batch_size=TEACHER_ENCODE_BATCH_SIZE,
            )
        )
    # Sub-evaluators alternate MSE / Translation per language: scores[1::2] are the translation accuracies.
    return SequentialEvaluator(sub_evaluators, main_score_function=lambda scores: float(np.mean(scores[1::2])))


def main() -> None:
    # [FUNCTION DOC]
    # Entry point. Parses only an optional --eval-only model path (unknown arguments, e.g. from
    # launchers, are ignored) and sets up logging. In eval-only mode it loads that saved model, runs
    # the per-language sequential evaluator (loading the teacher and the eval data) under autocast and returns.
    # Otherwise: loads the teacher and the student (multilingual model card,
    # max_seq_length 128), appends Normalize to the student if only the teacher has one, exits if the
    # dimensions differ, loads the parallel data (trimmed to 50/20 rows per language in smoke tests)
    # and maps teacher labels onto every subset in batches of 10,000, removing the original columns.
    # Uses MSELoss over the multi-dataset dict, the sequential evaluator's "sequential_score" for the
    # baseline, selection and verdict. Training: 3 epochs, batch size 64, learning rate 2e-5.
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
    # KNOWN ISSUE: the dimension-mismatch message refers to a 'MISMATCHED EMBEDDING DIMS' section in the distillation script, whose docstring section is actually titled 'Mismatched dims'.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only", type=str, default=None, help="Skip training; load this saved model and run only the evaluator."
    )
    cli, _ = parser.parse_known_args()

    setup_logging()

    if cli.eval_only:
        logging.info(f"Eval-only mode: loading model from {cli.eval_only}")
        student = SentenceTransformer(cli.eval_only)
        teacher = SentenceTransformer(TEACHER_MODEL_NAME)
        _, eval_dict = load_parallel_data()
        evaluator = build_evaluator(eval_dict, teacher)
        with autocast_ctx():
            evaluator(student)
        return

    logging.info(f"Loading teacher: {TEACHER_MODEL_NAME}")
    teacher = SentenceTransformer(TEACHER_MODEL_NAME)

    logging.info(f"Loading student: {STUDENT_MODEL_NAME}")
    student = SentenceTransformer(
        STUDENT_MODEL_NAME,
        model_card_data=SentenceTransformerModelCardData(
            language=[SOURCE_LANGUAGE, *TARGET_LANGUAGES],
            license="apache-2.0",
            model_name=f"{STUDENT_MODEL_NAME.split('/')[-1]} multilingual from {TEACHER_MODEL_NAME.split('/')[-1]}",
        ),
    )
    student.max_seq_length = STUDENT_MAX_SEQ_LENGTH
    # Match the teacher's final Normalize. MSELoss against unit-norm targets fights student
    # outputs at norm ~5-10 and can silently regress
    if any(isinstance(m, Normalize) for m in teacher) and not any(isinstance(m, Normalize) for m in student):
        student.append(Normalize())

    if student.get_embedding_dimension() != teacher.get_embedding_dimension():
        raise SystemExit(
            f"Student dim ({student.get_embedding_dimension()}) != teacher dim "
            f"({teacher.get_embedding_dimension()}). MSELoss requires matching dims. "
            "Either pick a student with matching dim, or add a Dense projection layer "
            "(see train_sentence_transformer_distillation_example.py 'MISMATCHED EMBEDDING DIMS')."
        )

    logging.info("Loading parallel data")
    train_dict, eval_dict = load_parallel_data()
    if SMOKE_TEST:
        logging.info("SMOKE_TEST=1: trimming each language subset; will run max_steps=1 and skip Hub push")
        train_dict = DatasetDict({k: v.select(range(min(50, len(v)))) for k, v in train_dict.items()})
        eval_dict = DatasetDict({k: v.select(range(min(20, len(v)))) for k, v in eval_dict.items()})

    def attach_teacher_label(batch):
        # [FUNCTION DOC]
        # HELPER: Batched map function: keeps the English and non-English columns and adds a "label"
        # column with the teacher's embeddings of the English sentences.
        return {
            "english": batch["english"],
            "non_english": batch["non_english"],
            "label": teacher.encode(batch["english"], batch_size=TEACHER_ENCODE_BATCH_SIZE, show_progress_bar=False),
        }

    column_names = list(train_dict.values())[0].column_names
    logging.info("Encoding training English sentences with teacher (cached on disk if you save_to_disk)")
    train_dict = train_dict.map(attach_teacher_label, batched=True, batch_size=10_000, remove_columns=column_names)
    eval_dict = eval_dict.map(attach_teacher_label, batched=True, batch_size=10_000, remove_columns=column_names)

    loss = MSELoss(model=student)

    evaluator = build_evaluator(eval_dict, teacher)
    logging.info("Student baseline (before training):")
    with autocast_ctx():
        baseline_eval = evaluator(student)["sequential_score"]

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        max_steps=1 if SMOKE_TEST else -1,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        learning_rate=2e-5,
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
        metric_for_best_model="eval_sequential_score",
        greater_is_better=True,
        report_to="none" if SMOKE_TEST else "trackio",
        run_name=RUN_NAME,
        seed=12,
    )

    trainer = SentenceTransformerTrainer(
        model=student,
        args=args,
        train_dataset=train_dict,
        eval_dataset=eval_dict,
        loss=loss,
        evaluator=evaluator,
    )
    if not SMOKE_TEST:
        log_trackio_dashboard()
    trainer.train()

    logging.info("Final student evaluation:")
    with autocast_ctx():
        score = evaluator(student)["sequential_score"]
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
