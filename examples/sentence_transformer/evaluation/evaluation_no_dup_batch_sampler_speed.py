# [MODULE DOC]
# Benchmark CLI for NoDuplicatesBatchSampler on a Hugging Face dataset, comparing the default sampler with the variant that
# precomputes text hashes. Optionally reports duplicate statistics and memory used while building hashes (tracemalloc, RSS or
# USS including child processes via psutil). Optional dependencies tqdm and psutil are imported if available. Disables the
# datasets cache globally at import. Defaults: msmarco co-condenser triplet-hard train split, batch size 8192, drop_last
# enabled, seed 42.
# KNOWN ISSUE: the descriptive string is placed after the future import, so it is not the module's docstring.
from __future__ import annotations

"""Benchmark NoDuplicates batch samplers on Hugging Face datasets.

Quick run:
    python examples/sentence_transformer/evaluation/evaluation_no_dup_batch_sampler_speed.py --target hashed

Run examples:
    python examples/sentence_transformer/evaluation/evaluation_no_dup_batch_sampler_speed.py \
        --dataset-name sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1 \
        --dataset-subset triplet-50 --dataset-split train --batch-size 128 --measure-hash-uss --no-progress-bar
    python examples/sentence_transformer/evaluation/evaluation_no_dup_batch_sampler_speed.py \
        --dataset-name sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1 \
        --dataset-subset triplet-50 --dataset-split train --batch-size 8192 --measure-hash-uss --no-progress-bar
    python examples/sentence_transformer/evaluation/evaluation_no_dup_batch_sampler_speed.py \
        --dataset-name sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1 \
        --dataset-subset triplet-hard --dataset-split train --batch-size 128 --measure-hash-uss --no-progress-bar
    python examples/sentence_transformer/evaluation/evaluation_no_dup_batch_sampler_speed.py \
        --dataset-name sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1 \
        --dataset-subset triplet-hard --dataset-split train --batch-size 8192 --measure-hash-uss --no-progress-bar
"""

import argparse
import asyncio
import gc
import os
import threading
import time
import tracemalloc

import datasets
import torch
from datasets import Dataset, load_dataset

from sentence_transformers.base.sampler import NoDuplicatesBatchSampler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

datasets.disable_caching()

DEFAULT_DATASET_NAME = "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1"
DEFAULT_DATASET_SUBSET = "triplet-hard"
DEFAULT_DATASET_SPLIT = "train"

BATCH_SIZE = 8192
DROP_LAST = True
SEED = 42


def run_sampler(
    name: str,
    sampler_cls,
    dataset: Dataset,
    batch_size: int,
    drop_last: bool,
    seed: int,
    warmup: bool,
    show_progress: bool,
    measure_hash_mem: bool,
    measure_hash_rss: bool,
    measure_hash_uss: bool,
    sampler_kwargs: dict[str, object] | None = None,
) -> tuple[float, int]:
    # [FUNCTION DOC]
    # Benchmarks one sampler configuration. Inputs: a display name, the sampler class, the dataset, batch size, drop_last, seed,
    # warmup flag, progress flag, three memory-measurement flags and extra sampler keyword arguments. Creates a seeded torch
    # generator and the sampler. If any memory measurement is requested and the sampler precomputes hashes, it garbage-collects,
    # starts the requested measurements, calls the sampler's private hash-building step while timing it, stops the measurements
    # and prints current and peak memory deltas with build time (printing "n/a" if psutil is missing); if hashing is not
    # precomputed it prints "n/a" lines instead. Optionally iterates the sampler once as warmup. Then times a full iteration,
    # counting batches, compares with the ideal batch count (floor or ceiling of rows over batch size depending on drop_last),
    # prints the result and returns the elapsed seconds and batch count.
    """Run one sampler and print timing + batch count."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = sampler_cls(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        generator=generator,
        seed=seed,
        **(sampler_kwargs or {}),
    )

    uss_sampler = None
    # Optionally precompute hashes and measure their memory cost.
    if measure_hash_mem or measure_hash_rss or measure_hash_uss:
        if getattr(sampler, "precompute_hashes", False):
            gc.collect()
            if measure_hash_mem:
                tracemalloc.start()
                start_current, start_peak = tracemalloc.get_traced_memory()
            rss_sampler = None
            if measure_hash_rss and psutil is not None:
                rss_sampler = _RssSampler()
                rss_sampler.start()
            elif measure_hash_rss and psutil is None:
                print(f"{name} hash_rss: n/a (psutil not available)")
            if measure_hash_uss and psutil is not None:
                uss_sampler = _UssSampler()
                uss_sampler.start()
            if measure_hash_uss and psutil is None:
                print(f"{name} hash_uss: n/a (psutil not available)")

            start = time.perf_counter()
            sampler._build_hashes()
            build_time = time.perf_counter() - start
            gc.collect()

            if rss_sampler is not None:
                rss_sampler.stop()
                rss_report = rss_sampler.report()
                print(
                    f"{name} hash_rss: current_delta={_format_bytes(rss_report.current_delta)}, "
                    f"peak_delta={_format_bytes(rss_report.peak_delta)}, build_time={build_time:.3f}s"
                )
            if uss_sampler is not None:
                uss_sampler.stop()
                uss_report = uss_sampler.report()
                print(
                    f"{name} hash_uss: current_delta={_format_bytes(uss_report.current_delta)}, "
                    f"peak_delta={_format_bytes(uss_report.peak_delta)}, build_time={build_time:.3f}s"
                )

            if measure_hash_mem:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                current_delta = current - start_current
                peak_delta = peak - start_peak
                print(
                    f"{name} hash_mem: current_delta={_format_bytes(current_delta)}, "
                    f"peak_delta={_format_bytes(peak_delta)}, build_time={build_time:.3f}s"
                )
        else:
            if measure_hash_rss:
                print(f"{name} hash_rss: n/a (precompute_hashes disabled)")
            if measure_hash_uss:
                print(f"{name} hash_uss: n/a (precompute_hashes disabled)")
            if measure_hash_mem:
                print(f"{name} hash_mem: n/a (precompute_hashes disabled)")

    # Warm up to reduce first-iteration overhead if requested.
    if warmup:
        warmup_iter = sampler
        if show_progress and tqdm is not None:
            warmup_iter = tqdm(warmup_iter, desc=f"{name} warmup", unit="batch")
        for _ in warmup_iter:
            pass

    # Timed pass.
    start = time.perf_counter()
    batch_count = 0
    timed_iter = sampler
    if show_progress and tqdm is not None:
        timed_iter = tqdm(timed_iter, desc=f"{name} timed", unit="batch")
    for _ in timed_iter:
        batch_count += 1
    elapsed = time.perf_counter() - start
    total_rows = len(dataset)
    ideal_batches = total_rows // batch_size if drop_last else (total_rows + batch_size - 1) // batch_size
    batch_delta = ideal_batches - batch_count
    print(
        f"{name}: {elapsed:.3f}s ({batch_count} batches; "
        f"ideal={ideal_batches}; delta={batch_delta}; batch_size={batch_size})"
    )
    return elapsed, batch_count


def parse_args() -> argparse.Namespace:
    # [FUNCTION DOC]
    # Defines and parses the command-line options: dataset name, subset and split, batch size, seed, warmup, progress-bar disable,
    # uniqueness statistics and their worker count (default 8), the three memory measurements, hashing process count and map batch
    # size, and a repeatable target option (default or hashed). Returns the parsed namespace; exits on invalid arguments.
    parser = argparse.ArgumentParser(description="Benchmark NoDuplicates batch samplers on Hugging Face datasets.")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME, help="Hugging Face dataset ID.")
    parser.add_argument(
        "--dataset-subset",
        type=str,
        default=DEFAULT_DATASET_SUBSET,
        help="Dataset subset/config name (if applicable).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=DEFAULT_DATASET_SPLIT,
        help="Dataset split to load (e.g. train/validation/test).",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for sampling order.")
    parser.add_argument("--warmup", action="store_true", help="Run a warmup pass before timing.")
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--show-uniqueness", action="store_true", help="Compute and display uniqueness stats.")
    parser.add_argument("--uniqueness-workers", type=int, default=8, help="Max worker threads for uniqueness stats.")
    parser.add_argument("--measure-hash-mem", action="store_true", help="Measure hash memory via tracemalloc.")
    parser.add_argument("--measure-hash-rss", action="store_true", help="Measure hash RSS via psutil.")
    parser.add_argument("--measure-hash-uss", action="store_true", help="Measure hash USS via psutil.")
    parser.add_argument("--precompute-num-proc", type=int, help="Processes used for hashing (hashed target only).")
    parser.add_argument(
        "--precompute-batch-size",
        type=int,
        help="datasets.map batch size for hashing (hashed target only).",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=["default", "hashed"],
        help="Which sampler to run (can be passed multiple times).",
    )
    return parser.parse_args()


def _iter_texts(value: object) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: returns a list of string conversions of the items if the value is a list, otherwise a one-element list with the
    # value converted to string.
    """Normalize a value into a list of strings for counting."""
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _format_bytes(value: int) -> str:
    # [FUNCTION DOC]
    # HELPER: formats a byte count as a string with two decimals in the largest unit (B, KiB, MiB, GiB, TiB) that keeps the absolute
    # value below 1024, capping at TiB.
    """Human-readable byte formatting."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}TiB"


class _RssReport:
    # [CLASS DOC]
    # HELPER: plain record of RSS measurements: start, end and peak values plus derived current delta (end minus start) and peak
    # delta (peak minus start).
    def __init__(self, start_rss: int, end_rss: int, peak_rss: int) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the three values and computes both deltas.
        self.start_rss = start_rss
        self.end_rss = end_rss
        self.peak_rss = peak_rss
        self.current_delta = end_rss - start_rss
        self.peak_delta = peak_rss - start_rss


class _RssSampler:
    # [CLASS DOC]
    # HELPER: background-thread sampler of the resident set size of this process and all its child processes, polling at a fixed
    # interval and tracking the start, end and peak totals.
    """Sample RSS (including child processes) during hashing."""

    def __init__(self, interval: float = 0.1) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the polling interval (default 0.1 seconds), creates a stop event, and initializes the thread to None and the
        # start, end and peak values to 0.
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_rss = 0
        self._end_rss = 0
        self._peak_rss = 0

    def _total_rss(self) -> int:
        # [FUNCTION DOC]
        # HELPER: returns the summed RSS in bytes of the current process and its recursive children, skipping children that vanished;
        # returns 0 if psutil is missing or the current process cannot be found.
        if psutil is None:
            return 0
        proc = psutil.Process(os.getpid())
        total = 0
        try:
            total += proc.memory_info().rss
        except psutil.NoSuchProcess:
            return 0
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        return total

    def _run(self) -> None:
        # [FUNCTION DOC]
        # HELPER: thread loop that samples total RSS, updates the peak, and sleeps for the interval until the stop event is set.
        while not self._stop_event.is_set():
            rss = self._total_rss()
            if rss > self._peak_rss:
                self._peak_rss = rss
            time.sleep(self.interval)

    def start(self) -> None:
        # [FUNCTION DOC]
        # HELPER: records the start RSS (also as initial peak) and starts the sampling loop on a daemon thread.
        self._start_rss = self._total_rss()
        self._peak_rss = self._start_rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # [FUNCTION DOC]
        # HELPER: signals the loop to stop, joins the thread, records the end RSS and raises the peak to it if larger.
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._end_rss = self._total_rss()
        if self._end_rss > self._peak_rss:
            self._peak_rss = self._end_rss

    def report(self) -> _RssReport:
        # [FUNCTION DOC]
        # HELPER: returns an _RssReport built from the recorded start, end and peak values.
        return _RssReport(self._start_rss, self._end_rss, self._peak_rss)


class _UssReport:
    # [CLASS DOC]
    # HELPER: plain record of USS (unique set size) measurements: start, end and peak plus current and peak deltas relative to start.
    def __init__(self, start_uss: int, end_uss: int, peak_uss: int) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the three values and computes both deltas.
        self.start_uss = start_uss
        self.end_uss = end_uss
        self.peak_uss = peak_uss
        self.current_delta = end_uss - start_uss
        self.peak_delta = peak_uss - start_uss


class _UssSampler:
    # [CLASS DOC]
    # HELPER: background-thread sampler of the unique set size of this process and its children, same structure as _RssSampler.
    """Sample USS (including child processes) during hashing."""

    def __init__(self, interval: float = 0.1) -> None:
        # [FUNCTION DOC]
        # HELPER: stores the interval (default 0.1 seconds), a stop event, no thread yet, and zeroed start, end and peak values.
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_uss = 0
        self._end_uss = 0
        self._peak_uss = 0

    def _total_uss(self) -> int:
        # [FUNCTION DOC]
        # HELPER: returns the summed USS of the current process and its recursive children from full memory info, skipping children
        # that vanished or lack USS support; returns 0 if psutil is missing or the current process cannot be inspected.
        if psutil is None:
            return 0
        proc = psutil.Process(os.getpid())
        total = 0
        try:
            total += proc.memory_full_info().uss
        except (psutil.NoSuchProcess, AttributeError):
            return 0
        for child in proc.children(recursive=True):
            try:
                total += child.memory_full_info().uss
            except (psutil.NoSuchProcess, AttributeError):
                continue
        return total

    def _run(self) -> None:
        # [FUNCTION DOC]
        # HELPER: thread loop that samples total USS, updates the peak and sleeps for the interval until stopped.
        while not self._stop_event.is_set():
            uss = self._total_uss()
            if uss > self._peak_uss:
                self._peak_uss = uss
            time.sleep(self.interval)

    def start(self) -> None:
        # [FUNCTION DOC]
        # HELPER: records the start USS as start and peak, then starts the daemon sampling thread.
        self._start_uss = self._total_uss()
        self._peak_uss = self._start_uss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # [FUNCTION DOC]
        # HELPER: stops and joins the thread, records the end USS and updates the peak if needed.
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._end_uss = self._total_uss()
        if self._end_uss > self._peak_uss:
            self._peak_uss = self._end_uss

    def report(self) -> _UssReport:
        # [FUNCTION DOC]
        # HELPER: returns a _UssReport of the recorded values.
        return _UssReport(self._start_uss, self._end_uss, self._peak_uss)


def _dup_stats(dataset: Dataset, show_progress: bool, desc: str) -> tuple[int, int, int]:
    # [FUNCTION DOC]
    # HELPER: counts text occurrences in a dataset. Picks the query column ("query", else "anchor"), document columns from a fixed
    # list of common names plus any column starting with "neg_" or "negative_", and if none were found but there are at least two
    # columns, uses the first as query and second as document. Iterates over all rows (with an optional progress bar), counting
    # each text (list values contribute each item). Returns the total count, the number of unique texts and the duplicates count.
    """Compute total/unique/dup counts across query/doc columns."""
    column_names = list(dataset.column_names)

    query_column = None
    if "query" in column_names:
        query_column = "query"
    elif "anchor" in column_names:
        query_column = "anchor"

    doc_columns = []
    doc_candidates = ["text", "positive", "pos", "doc", "document", "negative"]
    for name in doc_candidates:
        if name in column_names:
            doc_columns.append(name)
    for name in column_names:
        if name.startswith("neg_") or name.startswith("negative_"):
            doc_columns.append(name)

    if query_column is None and not doc_columns and len(column_names) >= 2:
        query_column = column_names[0]
        doc_columns = [column_names[1]]

    counts: dict[str, int] = {}
    row_iter = dataset
    if show_progress and tqdm is not None:
        row_iter = tqdm(row_iter, desc=f"uniqueness:{desc}", unit="row")
    for row in row_iter:
        if query_column is not None:
            for text in _iter_texts(row.get(query_column)):
                counts[text] = counts.get(text, 0) + 1
        for doc_column in doc_columns:
            for text in _iter_texts(row.get(doc_column)):
                counts[text] = counts.get(text, 0) + 1

    total = sum(counts.values())
    unique = len(counts)
    dup = total - unique
    return total, unique, dup


async def compute_uniqueness(
    datasets_map: dict[str, Dataset], workers: int, show_progress: bool
) -> dict[str, tuple[int, int, int] | None]:
    # [FUNCTION DOC]
    # Asynchronously computes duplicate statistics for several named datasets, running each _dup_stats call in a worker thread,
    # with at most the given number running concurrently via a semaphore. Returns a dict from name to the stats tuple, or None
    # for datasets whose computation raised an exception.
    """Run uniqueness checks concurrently with a bounded thread pool."""
    semaphore = asyncio.Semaphore(workers)
    results: dict[str, tuple[int, int, int] | None] = {}

    async def run_one(name: str, dataset: Dataset) -> tuple[str, tuple[int, int, int] | None]:
        # [FUNCTION DOC]
        # HELPER: under the semaphore, runs _dup_stats in a thread for one dataset and returns the name with its stats, or with None if
        # it raised.
        async with semaphore:
            try:
                stats = await asyncio.to_thread(_dup_stats, dataset, show_progress, name)
            except Exception:
                return name, None
            return name, stats

    tasks = [asyncio.create_task(run_one(name, dataset)) for name, dataset in datasets_map.items()]
    for name, stats in await asyncio.gather(*tasks):
        results[name] = stats

    return results


def _load_hf_dataset(name: str, subset: str | None, split: str) -> Dataset:
    # [FUNCTION DOC]
    # HELPER: loads the given split of a Hugging Face dataset, passing the subset as configuration name only when it is non-empty.
    # Returns the Dataset.
    """Load a HF dataset split with an optional subset/config."""
    if subset:
        return load_dataset(name, subset, split=split)
    return load_dataset(name, split=split)


def main() -> None:
    # [FUNCTION DOC]
    # Parses arguments, loads the dataset (an empty subset means none) and prints the benchmark settings and row count. If
    # uniqueness stats are requested, computes them via asyncio and prints totals and duplicate rate, or "failed". Then, for each
    # requested target (default both), runs run_sampler with the plain sampler or the hash-precomputing sampler configured with the
    # optional hashing process count and batch size. Returns None.
    args = parse_args()
    dataset_subset = args.dataset_subset or None
    dataset = _load_hf_dataset(args.dataset_name, dataset_subset, args.dataset_split)
    dataset_key = f"hf_{args.dataset_name}_{dataset_subset or 'default'}_{args.dataset_split}"

    print("Benchmark settings:")
    print(f"  batch_size={args.batch_size}, drop_last={DROP_LAST}, seed={args.seed}")
    print(f"  hf_dataset={args.dataset_name} subset={dataset_subset or 'default'} split={args.dataset_split}")
    print(f"  rows={len(dataset)}")

    if args.show_uniqueness:
        results = asyncio.run(
            compute_uniqueness(
                {dataset_key: dataset},
                workers=args.uniqueness_workers,
                show_progress=not args.no_progress_bar,
            )
        )
        stats = results.get(dataset_key)
        if stats is None:
            print("  uniqueness: failed")
        else:
            total, unique, dup = stats
            dup_rate = dup / total if total else 0.0
            print(f"  uniqueness: total={total} unique={unique} dup={dup} dup_rate={dup_rate:.6f}")

    targets = args.target or ["default", "hashed"]
    hashed_kwargs = {"precompute_hashes": True}
    if args.precompute_num_proc is not None:
        hashed_kwargs["precompute_num_proc"] = args.precompute_num_proc
    if args.precompute_batch_size is not None:
        hashed_kwargs["precompute_batch_size"] = args.precompute_batch_size
    target_map = {
        "default": ("NoDuplicatesBatchSampler", NoDuplicatesBatchSampler, {}),
        "hashed": ("NoDuplicatesBatchSampler (hashed)", NoDuplicatesBatchSampler, hashed_kwargs),
    }
    for target in targets:
        name, sampler_cls, sampler_kwargs = target_map[target]
        run_sampler(
            name,
            sampler_cls,
            dataset,
            args.batch_size,
            DROP_LAST,
            args.seed,
            warmup=args.warmup,
            show_progress=not args.no_progress_bar,
            measure_hash_mem=args.measure_hash_mem,
            measure_hash_rss=args.measure_hash_rss,
            measure_hash_uss=args.measure_hash_uss,
            sampler_kwargs=sampler_kwargs,
        )


if __name__ == "__main__":
    main()
