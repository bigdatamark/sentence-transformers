#!/usr/bin/env python3
# [MODULE DOC]
# Example script: trains a StaticEmbedding (token-bag, no transformer) SentenceTransformer on
# GooAQ pairs with MatryoshkaLoss over MultipleNegativesRankingLoss, with either random
# initialization from a BERT tokenizer (1024 dimensions, default) or a model2vec warm start
# (WARMSTART), using very large batches and high learning rates.
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
#     "tokenizers>=0.20",
#     "model2vec",  # only needed when WARMSTART=True (StaticEmbedding.from_model2vec)
# ]
# ///
"""Train a StaticEmbedding model on a contrastive dataset.

StaticEmbedding is a token-bag model: a per-token embedding table averaged over
the tokens of an input. No transformer, no attention. Inference is ~20x faster
on GPU and ~80x faster on CPU than a small encoder, with surprisingly competitive
quality on retrieval benchmarks when trained on >=1M contrastive pairs.

Two init paths via `WARMSTART` constant:
- `WARMSTART=False` (default): random init. Use with >=1M contrastive pairs.
  Reaches a higher ceiling than warm-start when given enough data. The default
  dataset below (GooAQ, ~3M pairs) is comfortably in this regime.
- `WARMSTART=True`: `StaticEmbedding.from_model2vec(...)` (distil from a
  model2vec checkpoint). Flip to True if you swap in a smaller dataset (<1M
  pairs). Converges faster and reaches better quality at lower data scales.

Demonstrates:
- MultipleNegativesRankingLoss wrapped in MatryoshkaLoss for nested embedding dims
- Large batch size (1024+) with a high LR (~2e-1 for random init, ~5e-2 for warm-
  start) since the loss surface for a token-bag is much flatter than for a
  pretrained encoder
- BatchSamplers.NO_DUPLICATES (load-bearing for in-batch negatives with duplicated
  anchors)
- NanoBEIREvaluator at full embedding dim
- Auto model card + optional Hub push

Run locally (CPU works for inference, but training needs a GPU for batch=1024+):
    pip install "sentence-transformers[train]>=5.0"
    python train_sentence_transformer_static_embedding_example.py

Multi-GPU:
    accelerate launch train_sentence_transformer_static_embedding_example.py

Hugging Face Jobs: paste this file's contents as the `script` in hf_jobs(...).

References:
- HF blog post: https://huggingface.co/blog/static-embeddings
- Module docs: sentence_transformers.sentence_transformer.modules.StaticEmbedding
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext

import datasets
import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator
from sentence_transformers.sentence_transformer.losses import (
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.modules import StaticEmbedding


def autocast_ctx():
    # [FUNCTION DOC]
    # HELPER: Returns a context manager for mixed precision around evaluator calls made outside the
    # trainer: a no-op context when CUDA is unavailable, otherwise CUDA autocast in bfloat16 if the GPU
    # supports it, else float16.
    """bf16/fp16 autocast for evaluator calls outside the trainer."""
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


TOKENIZER_NAME = "google-bert/bert-base-uncased"
EMBEDDING_DIM = 1024
MATRYOSHKA_DIMS = [1024, 512, 256, 128, 64, 32]  # ordered largest-first per MatryoshkaLoss

# False: random init (recommended for >=1M pairs, reaches a higher ceiling).
# True: warm-start from a model2vec checkpoint (recommended for <1M pairs).
# Default False because the example dataset (GooAQ, ~3M pairs) is well above the threshold.
WARMSTART = False
WARMSTART_MODEL2VEC = "minishlab/potion-base-8M"

OUTPUT_DIR = "models/static-embedding-bert-uncased"
RUN_NAME = "static-embedding-bert-uncased"
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
        torch.set_float32_matmul_precision("high")


def load_pair_dataset() -> datasets.Dataset:
    # [FUNCTION DOC]
    # HELPER: Loads the GooAQ train split, renames question/answer to anchor/positive and keeps only those two columns. Returns the dataset.
    """Load a contrastive-pair dataset for training.

    StaticEmbedding starts from random initialization, so it needs *a lot* of
    contrastive signal to converge. GooAQ alone provides ~3M (question, answer)
    pairs, comfortably over the >=1M threshold below which a warm-start would
    beat random init. For stronger production models, concatenate more sources
    (NaturalQuestions, MSMARCO, MIRACL, etc.) and shuffle, in the same family of
    sources used in `sentence-transformers/static-retrieval-mrl-en-v1`.
    """
    return (
        load_dataset("sentence-transformers/gooaq", split="train")
        .rename_columns({"question": "anchor", "answer": "positive"})
        .select_columns(["anchor", "positive"])
    )


def main() -> None:
    # [FUNCTION DOC]
    # Entry point. Parses only an optional --eval-only model path (unknown arguments, e.g. from
    # launchers, are ignored) and sets up logging. In eval-only mode it loads that saved model, runs
    # a 3-dataset NanoBEIR evaluator (msmarco, nfcorpus, nq) under autocast and returns.
    # Otherwise: builds the StaticEmbedding module (from the model2vec checkpoint if
    # WARMSTART, else randomly initialized over the tokenizer's vocabulary with EMBEDDING_DIM) and wraps
    # it in a SentenceTransformer with a model card. Loads the pairs (first 200 in smoke tests), splits
    # off eval rows (20 or 10,000, seed 12), builds the Matryoshka loss over MATRYOSHKA_DIMS. Training:
    # batch size 2048, learning rate 0.05 (warm start) or 0.2 (random), no weight decay, linear schedule,
    # no-duplicates sampler.
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
    # KNOWN ISSUE: with WARMSTART enabled the model2vec checkpoint has its own (smaller) embedding size, but MATRYOSHKA_DIMS (starting at 1024) and the model card name still assume EMBEDDING_DIM, so the Matryoshka dimensions exceed the real embedding size.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only", type=str, default=None, help="Skip training; load this saved model and run only the evaluator."
    )
    cli, _ = parser.parse_known_args()

    setup_logging()

    if cli.eval_only:
        logging.info(f"Eval-only mode: loading model from {cli.eval_only}")
        model = SentenceTransformer(cli.eval_only)
        evaluator = NanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"])
        with autocast_ctx():
            evaluator(model)
        return

    if WARMSTART:
        logging.info(f"Warm-starting StaticEmbedding from model2vec: {WARMSTART_MODEL2VEC}")
        # `StaticEmbedding.from_distillation("<bi-encoder>", vocabulary=...)` is the
        # alternative warm-start path (distil from a stronger teacher's vectors). Pick
        # one. model2vec is faster to load and converges quickly on smaller datasets.
        static_embedding = StaticEmbedding.from_model2vec(WARMSTART_MODEL2VEC)
    else:
        logging.info(f"Random-init StaticEmbedding from {TOKENIZER_NAME} tokenizer (dim={EMBEDDING_DIM})")
        tokenizer = Tokenizer.from_pretrained(TOKENIZER_NAME)
        static_embedding = StaticEmbedding(tokenizer, embedding_dim=EMBEDDING_DIM)
    model = SentenceTransformer(
        modules=[static_embedding],
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"Static embedding ({EMBEDDING_DIM}d) trained on contrastive pairs",
        ),
    )

    logging.info("Loading + concatenating training datasets")
    full = load_pair_dataset()
    if SMOKE_TEST:
        logging.info("SMOKE_TEST=1: trimmed dataset; will run max_steps=1 and skip Hub push")
        full = full.select(range(min(200, len(full))))
    eval_size = 20 if SMOKE_TEST else 10_000
    split = full.train_test_split(test_size=eval_size, seed=12)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logging.info(f"  train: {len(train_dataset):,} rows | eval: {len(eval_dataset):,} rows")
    logging.info(f"  columns: {train_dataset.column_names}")

    inner = MultipleNegativesRankingLoss(model)
    loss = MatryoshkaLoss(model, inner, matryoshka_dims=MATRYOSHKA_DIMS)

    evaluator = NanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"])
    logging.info("Baseline evaluation (random init scores near zero; warm-start scores 0.3+):")
    with autocast_ctx():
        # Must run before deriving metric_key: evaluator(model) mutates primary_metric to add the name_ prefix.
        baseline_eval = evaluator(model)[evaluator.primary_metric]
    metric_key = f"eval_{evaluator.primary_metric}"

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=1 if SMOKE_TEST else -1,
        per_device_train_batch_size=2048,
        per_device_eval_batch_size=2048,
        learning_rate=5e-2
        if WARMSTART
        else 2e-1,  # warm-start needs less LR. Both far higher than encoder fine-tuning
        weight_decay=0.0,  # weight decay on a token-bag is usually harmful
        warmup_steps=0.1,
        lr_scheduler_type="linear",
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
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
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    if not SMOKE_TEST:
        log_trackio_dashboard()
    trainer.train()

    logging.info("Post-training evaluation:")
    with autocast_ctx():
        score = evaluator(model)[evaluator.primary_metric]
    delta = score - baseline_eval
    verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
    logging.info(f"VERDICT: {verdict} | score={score:.4f} | baseline={baseline_eval:.4f} | delta={delta:+.4f}")

    final_dir = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final_dir)
    logging.info(f"Saved final model to {final_dir}")

    if SMOKE_TEST:
        logging.info("SMOKE_TEST=1: skipping Hub push")
        return

    try:
        commit_url = model.push_to_hub(RUN_NAME)
        logging.info(f"Pushed model to {commit_url.rsplit('/commit/', 1)[0]}")
    except Exception:
        import traceback

        logging.error(f"Hub push failed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
