#!/usr/bin/env python3
# [MODULE DOC]
# Example script: trains a multi-vector (ColBERT-style, late interaction) MultiVectorEncoder
# from ModernBERT-base on MS MARCO BM25 triplets with MultiVectorMultipleNegativesRankingLoss
# (MaxSim scoring), using the no-duplicates batch sampler and MultiVectorNanoBEIR evaluators.
# Module-level constants name the base model(s), datasets, sizes, output directory and run name;
# SMOKE_TEST is true when the SMOKE_TEST environment variable equals "1" and shrinks data to a few
# rows, limits training to one step and skips the Hub push. Running the file as a script calls main.
# The inline script metadata block lists dependencies for "uv run".
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers[train]>=6.0",
#     "datasets>=2.19.0",
#     "accelerate>=0.26.0",
#     "trackio",
# ]
# ///
"""Production-ready multi-vector (ColBERT / late-interaction) training template.

This script demonstrates a recommended setup:
- MultiVectorMultipleNegativesRankingLoss on (anchor, positive, negative) triplets, scored with MaxSim
- MultiVectorNanoBEIREvaluator for retrieval metrics during training
- BatchSamplers.NO_DUPLICATES (critical for MNRL-family)
- load_best_model_at_end with a retrieval metric
- Auto model card + optional Hub push

Runs identically in two modes:

    # Local
    pip install "sentence-transformers[train]>=6.0"
    python train_multi_vector_encoder_example.py

    # Or with uv (no explicit install needed)
    uv run train_multi_vector_encoder_example.py

    # Multi-GPU
    accelerate launch train_multi_vector_encoder_example.py

    # Hugging Face Jobs (paste the entire file contents as `script`)
    hf_jobs("uv", {
        "script": "<contents of this file>",
        "flavor": "a10g-large",
        "timeout": "3h",
        "secrets": {"HF_TOKEN": "$HF_TOKEN"},
    })

Adjust MODEL_NAME, DATASET_NAME, OUTPUT_DIR, RUN_NAME at the top of the script.
Default Hub push: at end of run, public, under your authenticated user as
`{user}/{RUN_NAME}`, wrapped in try/except. To skip the push, comment out the
push_to_hub call. For HF Jobs (ephemeral env), also enable in-trainer push:
add `push_to_hub=True`, `hub_model_id=RUN_NAME`, `hub_strategy="every_save"`
to TrainingArguments.
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext

import torch
from datasets import load_dataset

from sentence_transformers import (
    MultiVectorEncoder,
    MultiVectorEncoderModelCardData,
    MultiVectorEncoderTrainer,
    MultiVectorEncoderTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator
from sentence_transformers.multi_vector_encoder.losses import MultiVectorMultipleNegativesRankingLoss


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


# Bare HF encoder. MVE grafts a fresh 128-dim token projection on top.
MODEL_NAME = "answerdotai/ModernBERT-base"
DATASET_NAME = "sentence-transformers/msmarco-bm25"
DATASET_SUBSET = "triplet"
TRAIN_SIZE = 50_000
EVAL_SIZE = 1_000
OUTPUT_DIR = "models/modernbert-base-msmarco-colbert"
RUN_NAME = "modernbert-base-msmarco-colbert"
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
    # the full MultiVectorNanoBEIR evaluator (batch size 16) under autocast and returns.
    # Otherwise: loads the base model in float32 (autocast handles bf16) with an
    # English model card, takes the first train+eval triplets (50+20 in smoke tests) and splits them
    # (seed 12), builds the multi-vector MNRL loss with default scale, a cheap 3-dataset evaluator
    # (msmarco, nq, fiqa2018) for training-time selection and a full-suite test evaluator. The baseline
    # runs both; the verdict compares the full-suite primary score before and after training.
    # Training: batch size 32, learning rate 3e-5, warmup 5 percent, no-duplicates sampler; no early
    # stopping.
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only", type=str, default=None, help="Skip training, load this saved model and run only the evaluator."
    )
    cli, _ = parser.parse_known_args()

    setup_logging()

    if cli.eval_only:
        logging.info(f"Eval-only mode: loading model from {cli.eval_only}")
        model = MultiVectorEncoder(cli.eval_only)
        evaluator = MultiVectorNanoBEIREvaluator(batch_size=16)
        with autocast_ctx():
            evaluator(model)
        return

    logging.info(f"Loading base model: {MODEL_NAME}")
    # Load in fp32, autocast bf16 during forward. torch_dtype=bfloat16 breaks the optimizer's precision.
    model = MultiVectorEncoder(
        MODEL_NAME,
        model_card_data=MultiVectorEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"ColBERT {MODEL_NAME.split('/')[-1]} trained on MS MARCO",
        ),
        model_kwargs={"torch_dtype": "float32"},
    )

    logging.info(f"Loading dataset: {DATASET_NAME} ({DATASET_SUBSET})")
    train_size = 50 if SMOKE_TEST else TRAIN_SIZE
    eval_size = 20 if SMOKE_TEST else EVAL_SIZE
    full_dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, split="train").select(range(train_size + eval_size))
    dataset_dict = full_dataset.train_test_split(test_size=eval_size, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    if SMOKE_TEST:
        logging.info("SMOKE_TEST=1: trimmed dataset, will run max_steps=1 and skip Hub push")
    logging.info(f"  train: {len(train_dataset):,} examples")
    logging.info(f"  eval:  {len(eval_dataset):,} examples")

    # MNRL with in-batch negatives + explicit hard negatives. scale=1.0 (default) is correct for
    # unnormalized MaxSim: do not copy scale=20.0 from bi-encoder MNRL. Length-normalized MeanMaxSim
    # scoring instead wants a scale of roughly the average query length.
    loss = MultiVectorMultipleNegativesRankingLoss(model=model)

    # Cheap in-training evaluator on 3 datasets (drives `load_best_model_at_end`).
    # The end-of-run evaluator below covers the full 13-dataset suite for the VERDICT delta.
    evaluator = MultiVectorNanoBEIREvaluator(dataset_names=["msmarco", "nq", "fiqa2018"], batch_size=16)
    test_evaluator = MultiVectorNanoBEIREvaluator(show_progress_bar=True, batch_size=16)
    logging.info("Baseline evaluation (before training):")
    with autocast_ctx():
        # Must run before deriving metric_key: evaluator(model) mutates primary_metric to add the name_ prefix.
        evaluator(model)  # populates primary_metric on the training-time evaluator
        # Baseline on the full suite so the VERDICT delta is apples-to-apples with the post-training score.
        baseline_eval = test_evaluator(model)[test_evaluator.primary_metric]
    metric_key = f"eval_{evaluator.primary_metric}"

    args = MultiVectorEncoderTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=1 if SMOKE_TEST else -1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=0.05,
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

    trainer = MultiVectorEncoderTrainer(
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

    logging.info("Post-training evaluation (full NanoBEIR suite):")
    with autocast_ctx():
        score = test_evaluator(model)[test_evaluator.primary_metric]
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
        commit_url = model.push_to_hub(RUN_NAME)  # public by default, uses your authenticated user
        logging.info(f"Pushed model to {commit_url.rsplit('/commit/', 1)[0]}")
    except Exception:
        import traceback

        logging.error(f"Hub push failed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
