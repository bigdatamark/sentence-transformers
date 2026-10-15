# [MODULE DOC]
# Maintenance script (not a test) that fills in "transformers_tiny_models.json", a JSON object mapping each
# transformers model architecture key (e.g. "bert", "llama") to the Hub id of a tiny random test model, or null.
# At import time it builds the module-level list of architecture keys from transformers' MODEL_MAPPING_NAMES,
# falling back to CONFIG_MAPPING_NAMES if the former cannot be imported.
# When run as a script: reads the JSON next to this file, adds every known architecture missing from it with a
# null value, sorts by key, queries the Hub for each null entry, overwrites the JSON (indent 2), and prints
# statistics (total, found, not found). Performs network calls and file writes.
from __future__ import annotations

import json
import time
from pathlib import Path

from huggingface_hub import list_models

try:
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

    MODEL_KEYS = list(MODEL_MAPPING_NAMES.keys())
except ImportError:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    MODEL_KEYS = list(CONFIG_MAPPING_NAMES.keys())


def find_model_for_architecture(architecture):
    # [FUNCTION DOC]
    # Searches the Hugging Face Hub for a tiny test model for one architecture. Inputs: the architecture key string.
    # Iterates over a fixed, ordered list of test-model authors (trl-internal-testing, tiny-random,
    # hf-internal-testing, hf-tiny-model-private, peft-internal-testing, onnx-internal-testing). For each author it
    # lists up to 100 Hub models tagged with the architecture. A model id matches when the architecture name, lowercased
    # with underscores and hyphens removed, is a substring of the similarly normalised model id, and additionally either
    # the id contains "tiny-random", or it contains "tiny" and the author is trl-internal-testing. The first author with
    # any matches wins: among its matches, return the shortest id whose final path segment ends with "Model" if any,
    # else the first match. Returns None if no author yields a match. Side effect: network calls to the Hub.
    # KNOWN ISSUE: The docstring only mentions hf-internal-testing and tiny-random, but six authors are searched (in a
    # different order), and "prefer the one ending with Model" actually picks the shortest such id.
    """
    Find a model from hf-internal-testing or tiny-random for the given architecture.
    If multiple models exist, prefer the one ending with 'Model'.
    """
    for author in [
        "trl-internal-testing",
        "tiny-random",
        "hf-internal-testing",
        "hf-tiny-model-private",
        "peft-internal-testing",
        "onnx-internal-testing",
    ]:
        # Search for models from the author with the architecture tag
        models = list(list_models(filter=architecture, author=author, limit=100))

        if not models:
            continue

        # Filter for models that match the architecture pattern
        matching_models = []
        for model in models:
            model_id = model.id
            # Check if the model_id contains the architecture name
            if architecture.lower().replace("_", "").replace("-", "") in model_id.lower().replace("_", "").replace(
                "-", ""
            ) and (
                "tiny-random" in model_id.lower() or "tiny" in model_id.lower() and author == "trl-internal-testing"
            ):
                matching_models.append(model_id)

        if not matching_models:
            continue

        # Prefer models ending with "Model"
        model_suffix_matches = [m for m in matching_models if m.split("/")[-1].endswith("Model")]
        if model_suffix_matches:
            # Return the shortest one that ends with "Model"
            return min(model_suffix_matches, key=len)

        # Otherwise return the first match
        return matching_models[0]

    return None


def update_tiny_model_mapping(tiny_model_mapping):
    # [FUNCTION DOC]
    # Produces a new mapping from an architecture-to-model-id mapping. Inputs: a dict of architecture key to model id
    # or None. Entries that already have a value are copied unchanged; for None entries it calls the Hub search above
    # and stores the found id or None. Prints a progress line per architecture and sleeps 0.2 seconds after each Hub
    # search to rate-limit. Returns the new dict (input is not mutated).
    """Update the tiny model mapping using the Hugging Face Hub API"""
    updated_mapping = {}
    total = len(tiny_model_mapping)

    for idx, (architecture, current_value) in enumerate(tiny_model_mapping.items(), 1):
        print(f"[{idx}/{total}] Processing {architecture}...", end=" ")

        # Skip if we already have a value
        if current_value is not None:
            updated_mapping[architecture] = current_value
            print(f"Skipped (already has value): {current_value}")
            continue

        # Try to find a model for this architecture
        found_model = find_model_for_architecture(architecture)

        if found_model:
            updated_mapping[architecture] = found_model
            print(f"Found: {found_model}")
        else:
            updated_mapping[architecture] = None
            print("Not found")

        time.sleep(0.2)

    return updated_mapping


if __name__ == "__main__":
    json_file = Path(__file__).parent / "transformers_tiny_models.json"
    with open(json_file, encoding="utf8") as f:
        tiny_model_mapping = json.load(f)

    # Add any missing architectures with None value
    for arch in sorted(set(MODEL_KEYS) - set(tiny_model_mapping.keys())):
        tiny_model_mapping[arch] = None
    tiny_model_mapping = dict(sorted(tiny_model_mapping.items()))

    updated_mapping = update_tiny_model_mapping(tiny_model_mapping)
    with open(json_file, "w", encoding="utf8") as f:
        json.dump(updated_mapping, f, indent=2)

    # Count statistics
    total = len(updated_mapping)
    found = sum(1 for v in updated_mapping.values() if v is not None)
    not_found = total - found
    print("Statistics:")
    print(f"  Total architectures: {total}")
    print(f"  Found models: {found}")
    print(f"  Not found: {not_found}")
