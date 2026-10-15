# [MODULE DOC]
# Evaluation script: runs the multi-vector NanoBEIR evaluator (all 13 datasets, batch 16) on the LateOn model.
"""Evaluate a pretrained multi-vector model on NanoBEIR.

NanoBEIR is a fast benchmarking suite of 13 small BEIR subsets, useful for quickly comparing models
without running the full BEIR evaluation. This script loads a model from the Hub and runs all 13
Nano-* IR datasets with MaxSim scoring.
"""

from __future__ import annotations

from pprint import pprint

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator


def main() -> None:
    # [FUNCTION DOC]
    # Loads the model, runs the evaluator, prints the primary metric value and pretty-prints every result key containing
    # "ndcg@10". Side effects: downloads, compute, printing. Returns None.
    model = MultiVectorEncoder("lightonai/LateOn")
    evaluator = MultiVectorNanoBEIREvaluator(batch_size=16)
    results = evaluator(model)
    print(f"Primary metric: {evaluator.primary_metric} = {results[evaluator.primary_metric]:.4f}")
    pprint({k: v for k, v in results.items() if "ndcg@10" in k})


if __name__ == "__main__":
    main()
