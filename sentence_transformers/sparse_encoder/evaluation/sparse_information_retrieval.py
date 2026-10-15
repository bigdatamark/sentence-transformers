# [MODULE DOC]
# Sparse version of the information retrieval evaluator: ranks a corpus for each query using
# sparse embeddings and reports accuracy, precision, recall, MRR, NDCG and MAP at various k, plus
# query and corpus sparsity statistics and an estimate of retrieval FLOPS.
from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import append_to_last_row, compute_count_vector

if TYPE_CHECKING:
    import numpy as np

    from sentence_transformers.sparse_encoder.model import SparseEncoder
    from sentence_transformers.util.similarity import SimilarityFunction


logger = logging.getLogger(__name__)


class SparseInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    # [CLASS DOC]
    # Subclass of InformationRetrievalEvaluator for SparseEncoder models. Adds max_active_dims,
    # sparsity_stats (separate query and corpus statistic lists), count_vectors (per side, the
    # summed per-dimension count of non-zero entries) and count_lengths (per side, the number of
    # texts per encode call). Retrieval metrics are inherited.
    """
    This evaluator extends :class:`~sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator` but is specifically designed for sparse encoder models.

    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)

    Args:
        queries (Dict[str, str]): A dictionary mapping query IDs to queries.
        corpus (Dict[str, str]): A dictionary mapping document IDs to documents.
        relevant_docs (Dict[str, Set[str]]): A dictionary mapping query IDs to a set of relevant document IDs.
        corpus_chunk_size (int): The size of each chunk of the corpus. Defaults to 50000.
        mrr_at_k (List[int]): A list of integers representing the values of k for MRR calculation. Defaults to [10].
        ndcg_at_k (List[int]): A list of integers representing the values of k for NDCG calculation. Defaults to [10].
        accuracy_at_k (List[int]): A list of integers representing the values of k for accuracy calculation. Defaults to [1, 3, 5, 10].
        precision_recall_at_k (List[int]): A list of integers representing the values of k for precision and recall calculation. Defaults to [1, 3, 5, 10].
        map_at_k (List[int]): A list of integers representing the values of k for MAP calculation. Defaults to [100].
        show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.
        batch_size (int): The batch size for evaluation. Defaults to 32.
        name (str): A name for the evaluation. Defaults to "".
        write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
        max_active_dims (Optional[int], optional): The maximum number of active dimensions to use.
            `None` uses the model's current `max_active_dims`. Defaults to None.
        score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to the ``similarity`` function from the ``model``.
        main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.
        query_prompt (str, optional): The prompt to be used when encoding the corpus. Defaults to None.
        query_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus. Defaults to None.
        corpus_prompt (str, optional): The prompt to be used when encoding the corpus. Defaults to None.
        corpus_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus. Defaults to None.
        write_predictions (bool): Whether to write the predictions to a JSONL file. Defaults to False.
            This can be useful for downstream evaluation as it can be used as input to the :class:`~sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator` that accept precomputed predictions.

    Example:
        ::

            import logging

            from datasets import load_dataset

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator

            logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

            # Load a model
            model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

            # Load the NFcorpus IR dataset (https://huggingface.co/datasets/BeIR/nfcorpus, https://huggingface.co/datasets/BeIR/nfcorpus-qrels)
            corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
            queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
            relevant_docs_data = load_dataset("BeIR/nfcorpus-qrels", split="test")

            # For this dataset, we want to concatenate the title and texts for the corpus
            corpus = corpus.map(lambda x: {"text": x["title"] + " " + x["text"]}, remove_columns=["title"])

            # Convert the datasets to dictionaries
            corpus = dict(zip(corpus["_id"], corpus["text"]))  # Our corpus (cid => document)
            queries = dict(zip(queries["_id"], queries["text"]))  # Our queries (qid => question)
            relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
            for qid, corpus_ids in zip(relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]):
                qid = str(qid)
                corpus_ids = str(corpus_ids)
                if qid not in relevant_docs:
                    relevant_docs[qid] = set()
                relevant_docs[qid].add(corpus_ids)

            # Given queries, a corpus and a mapping with relevant documents, the SparseInformationRetrievalEvaluator computes different IR metrics.
            ir_evaluator = SparseInformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name="BeIR-nfcorpus-subset-test",
                show_progress_bar=True,
                batch_size=16,
            )

            # Run evaluation
            results = ir_evaluator(model)
            '''
            Queries: 323
            Corpus: 3269

            Score-Function: dot
            Accuracy@1: 49.23%
            Accuracy@3: 63.47%
            Accuracy@5: 66.56%
            Accuracy@10: 71.83%
            Precision@1: 49.23%
            Precision@3: 39.63%
            Precision@5: 33.13%
            Precision@10: 25.23%
            Recall@1: 6.08%
            Recall@3: 11.60%
            Recall@5: 13.47%
            Recall@10: 17.01%
            MRR@10: 0.5706
            NDCG@10: 0.3530
            MAP@100: 0.1778
            Model Query Sparsity: Active Dimensions: 40.0, Sparsity Ratio: 0.9987
            Model Corpus Sparsity: Active Dimensions: 206.2, Sparsity Ratio: 0.9932
            Average FLOPS: 4.66
            '''
            # Print the results
            print(f"Primary metric: {ir_evaluator.primary_metric}")
            # => Primary metric: BeIR-nfcorpus-subset-test_dot_ndcg@10
            print(f"Primary metric value: {results[ir_evaluator.primary_metric]:.4f}")
            # => Primary metric value: 0.3530

    """

    def __init__(
        self,
        queries: dict[str, str],  # qid => query
        corpus: dict[str, str],  # cid => doc
        relevant_docs: dict[str, set[str]],  # qid => Set[cid]
        corpus_chunk_size: int = 50000,
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        max_active_dims: int | None = None,
        score_functions: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None = None,
        main_score_function: str | SimilarityFunction | None = None,
        query_prompt: str | None = None,
        query_prompt_name: str | None = None,
        corpus_prompt: str | None = None,
        corpus_prompt_name: str | None = None,
        write_predictions: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Stores max_active_dims, initializes empty sparsity/count state, and delegates queries, corpus,
        # relevant docs, chunk size, k lists, progress bar, batch size, name, CSV flag, score functions,
        # main score function, query/corpus prompts and prompt names and write_predictions to the parent.
        self.max_active_dims = max_active_dims
        self.sparsity_stats = {"query": defaultdict(list), "corpus": defaultdict(list)}
        self.count_vectors = {}
        self.count_lengths = {"query": [], "corpus": []}
        return super().__init__(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            corpus_chunk_size=corpus_chunk_size,
            mrr_at_k=mrr_at_k,
            ndcg_at_k=ndcg_at_k,
            accuracy_at_k=accuracy_at_k,
            precision_recall_at_k=precision_recall_at_k,
            map_at_k=map_at_k,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            name=name,
            write_csv=write_csv,
            score_functions=score_functions,
            main_score_function=main_score_function,
            query_prompt=query_prompt,
            query_prompt_name=query_prompt_name,
            corpus_prompt=corpus_prompt,
            corpus_prompt_name=corpus_prompt_name,
            write_predictions=write_predictions,
        )

    def _append_csv_headers(self, score_function_names):
        # [FUNCTION DOC]
        # Extends the parent's CSV headers; only when score_function_names is non-empty (so they are
        # added once) appends query_active_dims, query_sparsity_ratio, corpus_active_dims,
        # corpus_sparsity_ratio and avg_flops.
        super()._append_csv_headers(score_function_names)
        # To avoid adding the sparse-specific headers multiple times, we only add them if the superclass will
        # add metric columns for the specified score functions
        if score_function_names:
            self.csv_headers.extend(
                [
                    "query_active_dims",
                    "query_sparsity_ratio",
                    "corpus_active_dims",
                    "corpus_sparsity_ratio",
                    "avg_flops",
                ]
            )

    def __call__(
        self, model: SparseEncoder, output_path: str | None = None, epoch: int = -1, steps: int = -1, *args, **kwargs
    ) -> dict[str, float]:
        # [FUNCTION DOC]
        # Runs the evaluation. Resets sparsity and count state, runs the parent evaluation (which encodes
        # via embed_inputs), then averages query statistics as a plain mean over encode calls and corpus
        # statistics as a mean weighted by the number of texts per call. Flattens them into keys like
        # query_active_dims, adds avg_flops from _calculate_avg_flops, adds all of these (prefixed with
        # name) to the metrics, stores the metrics in the model card data, logs the query and corpus
        # sparsity and the FLOPS, and appends the values to the last CSV row when output_path is given
        # and write_csv is set. Extra args and kwargs are ignored. Returns the metrics dict.
        # KNOWN ISSUE: Query statistics use an unweighted mean over encode calls while corpus statistics
        # are weighted by batch size, so query values are skewed when encode calls have different sizes.
        self.sparsity_stats = {"query": defaultdict(list), "corpus": defaultdict(list)}
        self.count_vectors = {}
        self.count_lengths = {"query": [], "corpus": []}
        metrics = super().__call__(model=model, output_path=output_path, epoch=epoch, steps=steps)
        for prefix in ["query", "corpus"]:
            for key, value in self.sparsity_stats[prefix].items():
                if prefix == "query":
                    self.sparsity_stats[prefix][key] = sum(value) / len(value)
                else:
                    self.sparsity_stats[prefix][key] = sum(
                        val * length for val, length in zip(value, self.count_lengths[prefix])
                    ) / sum(self.count_lengths[prefix])

        self.sparsity_stats = {
            f"{prefix}_{key}": value for prefix, values in self.sparsity_stats.items() for key, value in values.items()
        }
        self.sparsity_stats["avg_flops"] = self._calculate_avg_flops()
        metrics.update(self.prefix_name_to_metrics(self.sparsity_stats, self.name))
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        logger.info(
            f"Model Query Sparsity: Active Dimensions: {self.sparsity_stats['query_active_dims']:.1f}, Sparsity Ratio: {self.sparsity_stats['query_sparsity_ratio']:.4f}"
        )
        logger.info(
            f"Model Corpus Sparsity: Active Dimensions: {self.sparsity_stats['corpus_active_dims']:.1f}, Sparsity Ratio: {self.sparsity_stats['corpus_sparsity_ratio']:.4f}"
        )
        logger.info(f"Average FLOPS: {self.sparsity_stats['avg_flops']:.2f}")
        if output_path is not None and self.write_csv:
            append_to_last_row(os.path.join(output_path, self.csv_file), self.sparsity_stats.values())

        return metrics

    def embed_inputs(
        self,
        model: SparseEncoder,
        sentences: str | list[str] | np.ndarray,
        encode_fn_name: str | None = None,
        prompt_name: str | None = None,
        prompt: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Encodes texts with model.encode (encode_fn_name None), encode_query ("query") or encode_document
        # ("document"), passing the prompt name/prompt, batch size, progress bar, max_active_dims and
        # extra kwargs, as sparse CPU tensors. Records sparsity statistics under the query side (for None
        # or "query") or the corpus side, adds the per-dimension non-zero count vector to that side's
        # running sum, and records the number of texts. Returns the embeddings.
        # KNOWN ISSUE: Any other encode_fn_name leaves the encode function unassigned and raises
        # UnboundLocalError; and a single string input counts its character length as the number of texts.
        if encode_fn_name is None:
            encode_fn = model.encode
        elif encode_fn_name == "query":
            encode_fn = model.encode_query
        elif encode_fn_name == "document":
            encode_fn = model.encode_document
        embeddings = encode_fn(
            sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
            max_active_dims=self.max_active_dims,
            **kwargs,
        )
        stat = model.sparsity(embeddings)
        prefix = "query" if encode_fn_name in ["query", None] else "corpus"
        for key, value in stat.items():
            self.sparsity_stats[prefix][key].append(value)
        count_vec = compute_count_vector(embeddings)
        if prefix in self.count_vectors:
            self.count_vectors[prefix] += count_vec
        else:
            self.count_vectors[prefix] = count_vec
        self.count_lengths[prefix].append(len(sentences))
        return embeddings

    def store_metrics_in_model_card_data(
        self, model: SparseEncoder, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        # [FUNCTION DOC]
        # Stores the metrics in the model's model card data for this evaluator at the given epoch and step.
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch=epoch, step=step)

    def get_config_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns the parent config dict plus max_active_dims when it is set.
        config_dict = super().get_config_dict()
        if self.max_active_dims is not None:
            config_dict["max_active_dims"] = self.max_active_dims
        return config_dict

    def _calculate_avg_flops(self) -> float:
        # [FUNCTION DOC]
        # HELPER: Returns the estimated FLOPS: the dot product of the average query count vector and the
        # average corpus count vector (each summed count vector divided by its number of texts), as a
        # float. Returns 0.0 if either side was never encoded.
        """Calculate average FLOPS between queries and corpus.

        FLOPS is the dot product of the average query and corpus count vectors,
        and can be used to estimate the computational cost of retrieval with
        sparse representations.
        """
        if "query" not in self.count_vectors or "corpus" not in self.count_vectors:
            return 0.0

        avg_query_count = self.count_vectors["query"] / sum(self.count_lengths["query"])
        avg_corpus_count = self.count_vectors["corpus"] / sum(self.count_lengths["corpus"])
        flops = float(torch.dot(avg_query_count, avg_corpus_count).cpu())
        return flops
