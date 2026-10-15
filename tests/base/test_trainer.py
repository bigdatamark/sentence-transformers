# [MODULE DOC]
# Pytest tests for SentenceTransformerTrainer internals: pushing checkpoints to the Hub (with Hub API calls mocked),
# reloading checkpoints that contain user-defined module classes, and accumulation of per-component loss values. Skips
# the module if the training extra is not installed. Uses the root conftest's static embedding fixtures and pytest
# temporary directories.
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from huggingface_hub import HfApi

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.modules import Normalize, Router
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.util import is_training_available

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


@contextlib.contextmanager
def _patch_hfapi(method: str, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Context manager that mocks a Hugging Face Hub API method by name. Older transformers imported such
    # functions into its trainer module, newer ones call the HfApi class method at call time; it patches the trainer
    # module attribute if present, else the HfApi class attribute, forwarding extra keyword arguments (e.g. return
    # value) to the mock. Yields the mock and restores the original on exit.
    # Older transformers imported create_repo/upload_folder into the trainer module.
    # Newer transformers resolves HfApi.<method> at call time via hf_api(), so patch
    # whichever the installed version actually uses.
    import transformers.trainer as hf_trainer

    target = hf_trainer if hasattr(hf_trainer, method) else HfApi
    with patch.object(target, method, **kwargs) as mock:
        yield mock


def test_push_from_checkpoint_copies_full_layout(static_embedding_model: SentenceTransformer, tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Creates a trainer with push to Hub enabled, a dummy hub model id and the "every_save" strategy (repository
    # creation mocked). Saves the model into a "checkpoint-437" folder, then adds dummy training-state files
    # (optimizer, scheduler, scaler, trainer state, RNG state), a DeepSpeed "global_step123" directory, and a
    # sharded-weights index with two shard files. Calls the private push-from-checkpoint method with the upload
    # mocked, the process marked as rank zero and the callback handler mocked. Asserts the Sentence Transformers
    # layout files (modules list, ST config, README, tokenizer) were copied into the output directory, none of the
    # training-state files or the DeepSpeed directory were, and the upload was called with the output directory as
    # folder path.
    output_dir = tmp_path / "out"
    checkpoint_folder = output_dir / "checkpoint-437"
    checkpoint_folder.mkdir(parents=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        push_to_hub=True,
        hub_model_id="dummy/model",
        hub_strategy="every_save",
        report_to=[],
    )

    with _patch_hfapi("create_repo", return_value=SimpleNamespace(repo_id="dummy/model")):
        trainer = SentenceTransformerTrainer(model=static_embedding_model, args=args)

    trainer._save(output_dir=str(checkpoint_folder))

    # Training-state files that _save_checkpoint would add on top of _save.
    for name in ("optimizer.pt", "scheduler.pt", "scaler.pt", "trainer_state.json", "rng_state.pth"):
        (checkpoint_folder / name).write_text("dummy")

    # DeepSpeed ZeRO shards, should not be pushed.
    global_step_dir = checkpoint_folder / "global_step123"
    global_step_dir.mkdir()
    (global_step_dir / "zero_state.bin").write_text("zero")

    # Sharded weights index: the shards listed inside must not be double-copied by our override.
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    (checkpoint_folder / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {f"layer.{i}.weight": s for i, s in enumerate(shard_names)}})
    )
    for shard in shard_names:
        (checkpoint_folder / shard).write_text("shard")

    with (
        _patch_hfapi("upload_folder") as mock_upload,
        patch.object(trainer, "is_world_process_zero", return_value=True),
        patch.object(trainer, "callback_handler"),
    ):
        trainer.push_in_progress = None
        trainer._push_from_checkpoint(str(checkpoint_folder))

    contents = set(os.listdir(output_dir))

    # Sentence Transformers layout files must reach output_dir via our override.
    for name in ("modules.json", "config_sentence_transformers.json", "README.md", "tokenizer.json"):
        assert name in contents, f"{name} missing from output_dir"

    # Training state and DeepSpeed dirs must never reach output_dir.
    for name in ("optimizer.pt", "scheduler.pt", "scaler.pt", "trainer_state.json", "rng_state.pth", "global_step123"):
        assert name not in contents, f"{name} should not be in output_dir"

    # Super still runs the actual upload on output_dir.
    assert mock_upload.called
    assert mock_upload.call_args.kwargs["folder_path"] == str(output_dir)


def test_push_from_checkpoint_skips_when_end_strategy(
    static_embedding_model: SentenceTransformer, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Same setup with the "end" hub strategy and a "checkpoint-1" folder; asserts the private push-from-checkpoint
    # method neither uploads nor copies the modules list into the output directory (pushing happens only at training
    # end).
    output_dir = tmp_path / "out"
    checkpoint_folder = output_dir / "checkpoint-1"
    checkpoint_folder.mkdir(parents=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        push_to_hub=True,
        hub_model_id="dummy/model",
        hub_strategy="end",
        report_to=[],
    )

    with _patch_hfapi("create_repo", return_value=SimpleNamespace(repo_id="dummy/model")):
        trainer = SentenceTransformerTrainer(model=static_embedding_model, args=args)

    trainer._save(output_dir=str(checkpoint_folder))

    with (
        _patch_hfapi("upload_folder") as mock_upload,
        patch.object(trainer, "is_world_process_zero", return_value=True),
        patch.object(trainer, "callback_handler"),
    ):
        trainer.push_in_progress = None
        trainer._push_from_checkpoint(str(checkpoint_folder))

    # hub_strategy="end" pushes only at the end of training, not mid-training. No copy, no upload.
    assert not mock_upload.called
    assert not (output_dir / "modules.json").exists()


class CustomNormalize(Normalize):
    # [CLASS DOC]
    # HELPER: Trivial subclass of the Normalize module defined in the tests namespace, standing in for a user's custom
    # module class whose import path is outside the library (and would normally require trust remote code).
    """A module class outside the ``sentence_transformers.*`` namespace, like a user's own custom module."""


@pytest.mark.parametrize("behind_router", [False, True])
def test_load_from_checkpoint_reloads_custom_module_classes(
    static_embedding: StaticEmbedding, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behind_router: bool
) -> None:
    # [FUNCTION DOC]
    # Parametrised on whether the custom module sits behind a Router (query and document routes both using it). Builds
    # a model from the static embedding plus the custom module (or router), with widget examples disabled, and asserts
    # trust remote code is off. Creates a trainer (with a router mapping of the text column to query when routed),
    # saves a checkpoint, and confirms the saved module type (from the modules list, or from the router config's last
    # route type) is not in the library namespace. Zeroes the first non-empty model parameter in place, then
    # monkeypatches the SentenceTransformer constructor to record its keyword arguments, calls the private
    # load-from-checkpoint method, and asserts the parameter is restored to its original values and the reload was
    # done with trust remote code still False.
    """Since v6.0, a module class outside Sentence Transformers requires ``trust_remote_code=True``, which a
    programmatically built model never carries. The trainer must still be able to reload its own checkpoint:
    otherwise ``load_best_model_at_end`` quietly keeps the final weights instead of the best ones, and
    ``resume_from_checkpoint`` raises. Handing over the classes it is already running is what gets it there,
    so the flag itself stays untouched and stale-remote-code backbones keep their native path. ``Router``
    resolves its own routes, so a custom class nested behind one has to be covered too."""
    custom = CustomNormalize()
    tail = Router({"query": [custom], "document": [custom]}) if behind_router else custom
    model = SentenceTransformer(modules=[static_embedding, tail])
    model.model_card_data.generate_widget_examples = False
    assert model.trust_remote_code is False

    args = SentenceTransformerTrainingArguments(
        output_dir=str(tmp_path / "out"),
        report_to=[],
        router_mapping={"text": "query"} if behind_router else None,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args)

    checkpoint_folder = tmp_path / "checkpoint-1"
    trainer._save(output_dir=str(checkpoint_folder))
    if behind_router:
        router_config = json.loads((checkpoint_folder / "1_Router" / "router_config.json").read_text())
        saved_type = list(router_config["types"].values())[-1]
    else:
        saved_type = json.loads((checkpoint_folder / "modules.json").read_text())[-1]["type"]
    assert not saved_type.startswith("sentence_transformers."), "the premise needs a non-ST module class"

    weights = next(parameter for parameter in model.parameters() if parameter.numel())
    original = weights.detach().clone()
    with torch.no_grad():
        weights.zero_()

    reload_kwargs = {}
    original_init = SentenceTransformer.__init__

    def recording_init(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Recording replacement for the SentenceTransformer constructor: merges the keyword arguments into the
        # captured dict and calls the original constructor.
        reload_kwargs.update(kwargs)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(SentenceTransformer, "__init__", recording_init)
    trainer._load_from_checkpoint(str(checkpoint_folder))

    assert torch.equal(weights, original)
    assert reload_kwargs["trust_remote_code"] is False


def test_track_loss_components_detaches_the_accumulated_values() -> None:
    # [FUNCTION DOC]
    # Calls the trainer's loss-component tracking method (as an unbound function) twice on a fake trainer namespace
    # (NaN/Inf filter off, training mode, empty train/eval accumulators, global step 0) with a graph-carrying tensor
    # of value 6. Asserts the accumulated "base_loss" value under "train" is detached (no gradient function, not
    # requiring grad) and equals 12.
    """The component dict values carry the autograd graph and, for the Cached* losses, the
    gradient caches held by their backward hook. Accumulating them undetached would pin
    every step's caches until the next logging flush (multi-GB at default logging_steps)."""
    trainer = SimpleNamespace(
        args=SimpleNamespace(logging_nan_inf_filter=False),
        model=SimpleNamespace(training=True),
        accum_loss_components={"train": {}, "eval": {}},
        state=SimpleNamespace(global_step=0),
        _globalstep_last_logged=0,
    )
    value = torch.tensor(2.0, requires_grad=True) * 3
    assert value.grad_fn is not None, "the test premise requires a graph-carrying component"

    SentenceTransformerTrainer.track_loss_components(trainer, {"base_loss": value})
    SentenceTransformerTrainer.track_loss_components(trainer, {"base_loss": value})

    accumulated = trainer.accum_loss_components["train"]["base_loss"]
    assert accumulated.grad_fn is None and not accumulated.requires_grad
    assert accumulated.item() == pytest.approx(12.0)
