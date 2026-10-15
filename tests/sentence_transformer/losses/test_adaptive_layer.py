# [MODULE DOC]
# Tests AdaptiveLayerLoss, which applies an inner loss to the final layer's embeddings and to embeddings from prior transformer layers
# (re-running pooling on them), weighted per layer, plus a KL self-distillation term from the final layer. Covers mask handling across
# repeated pooling passes, unwrapping the model from DDP/torch.compile wrappers (including nested and unknown ones), detaching the KL
# teacher, per-layer weighting schemes and their config reporting. Defines fake wrapper modules and a stub trainer.
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.trainer import BaseTrainer
from sentence_transformers.sentence_transformer.losses import AdaptiveLayerLoss, MultipleNegativesRankingLoss


class _FakeDDP(nn.Module):
    # [CLASS DOC]
    # HELPER: Fake DistributedDataParallel wrapper: stores the wrapped module as "module", forwards calls to it, and is not subscriptable.
    """Stand-in for `torch.nn.parallel.DistributedDataParallel` that exposes the wrapped
    module via `.module` and proxies `forward` like real DDP. Notably it does NOT support
    `__getitem__`, so `model[0]` raises TypeError."""

    def __init__(self, module: nn.Module) -> None:
        # [FUNCTION DOC]
        # HELPER: Initializes the module and registers the wrapped module.
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC]
        # HELPER: Forwards all arguments to the wrapped module and returns its output.
        return self.module(*args, **kwargs)


class _FakeCompile(nn.Module):
    # [CLASS DOC]
    # HELPER: Fake torch.compile output: stores the wrapped module as "_orig_mod", forwards calls to it, and is not subscriptable.
    """Stand-in for `torch._dynamo.OptimizedModule` (output of `torch.compile`): exposes
    the wrapped module via `_orig_mod` and proxies `forward`. Also not subscriptable."""

    def __init__(self, module: nn.Module) -> None:
        # [FUNCTION DOC]
        # HELPER: Initializes the module and registers the wrapped module as _orig_mod.
        super().__init__()
        self._orig_mod = module

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC]
        # HELPER: Forwards all arguments to the wrapped module.
        return self._orig_mod(*args, **kwargs)


class _StubTrainer:
    # [CLASS DOC]
    # HELPER: Minimal object that borrows BaseTrainer's override_model_in_loss method (which rebinds a loss's model attribute, recursively
    # for nested losses, and copies model methods onto wrappers) without building a full trainer.
    """Borrow `BaseTrainer.override_model_in_loss` without instantiating the full trainer.
    The method only uses `self` for its recursive call, so any object with the same bound
    attribute works."""

    override_model_in_loss = BaseTrainer.override_model_in_loss


def _make_loss(model: SentenceTransformer) -> AdaptiveLayerLoss:
    # [FUNCTION DOC]
    # HELPER: Returns an AdaptiveLayerLoss wrapping MultipleNegativesRankingLoss for the model, with default settings.
    inner_loss = MultipleNegativesRankingLoss(model)
    return AdaptiveLayerLoss(model, inner_loss)


def _features_and_labels(model: SentenceTransformer) -> tuple[list[dict], torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Preprocesses two columns of two short texts each, moves tensors to the model device, and returns them with labels 0 and 1.
    features = [
        {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}
        for feats in [model.preprocess(["a", "b"]), model.preprocess(["c", "d"])]
    ]
    labels = torch.tensor([0, 1], device=model.device)
    return features, labels


def test_adaptive_layer_loss_prior_layers_see_the_collated_mask(stsb_bert_tiny_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Disables prompt inclusion in pooling, preprocesses two columns with a "query: " prompt (asserting prompt_length is present), and runs
    # AdaptiveLayerLoss over all prior layers twice without gradients on the same feature dicts. Asserts both loss values are equal, which
    # holds only if pooling does not write a narrowed attention mask back into the shared features. Mutates the (per-test) model's pooling setting.
    """Each prior-layer pass re-runs the pooling half over the same feature dicts. If pooling wrote
    its prompt-excluded mask back into them, every pass after the first would narrow an already
    narrowed mask, so with ``include_prompt`` disabled the dicts have to keep the mask as collated.
    Pinned as loss equality across repeated calls, which only holds if they do."""
    model = stsb_bert_tiny_model
    model.set_pooling_include_prompt(False)
    features = [
        {key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in feats.items()}
        for feats in [
            model.preprocess(["It is sunny today.", "He drove to work."], prompt="query: "),
            model.preprocess(["The weather is lovely.", "He took the car."], prompt="query: "),
        ]
    ]
    assert all("prompt_length" in feature for feature in features), "the premise needs a prompt to exclude"

    adaptive = AdaptiveLayerLoss(model, MultipleNegativesRankingLoss(model), n_layers_per_step=-1)
    with torch.no_grad():
        first = adaptive(features, None).item()
        second = adaptive(features, None).item()
    assert second == pytest.approx(first, rel=1e-4, abs=1e-6)


def test_adaptive_layer_loss_runs_without_wrapper(stsb_bert_tiny_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts the loss on an unwrapped model is finite.
    """Sanity check for the non-DDP, non-compile path: a bare SentenceTransformer skips
    the unwrap loop entirely."""
    adaptive = _make_loss(stsb_bert_tiny_model)
    features, labels = _features_and_labels(stsb_bert_tiny_model)
    loss = adaptive(features, labels)
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "wrapper_cls",
    [_FakeDDP, _FakeCompile],
    ids=["ddp", "torch_compile"],
)
def test_adaptive_layer_loss_unwraps_wrapped_model(
    stsb_bert_tiny_model: SentenceTransformer,
    wrapper_cls: type[nn.Module],
) -> None:
    # [FUNCTION DOC]
    # Parametrized over the fake DDP and compile wrappers. Asserts indexing the wrapper raises TypeError, rebinds the loss's model to the
    # wrapper via the trainer method (asserting the model is the wrapper and gained a preprocess attribute), then asserts the loss is a
    # finite scalar, proving the loss unwraps to the inner model.
    """AdaptiveLayerLoss.forward reaches into `self.model[0]`. Under DDP / torch.compile
    the trainer rebinds `loss.model` to the wrapper (and binds BaseModel methods like
    `preprocess` onto it). Verify the loss still unwraps to the inner BaseModel for the
    `model[0]` decoration. See #3170.
    """
    adaptive = _make_loss(stsb_bert_tiny_model)
    wrapped = wrapper_cls(stsb_bert_tiny_model)
    with pytest.raises(TypeError):
        _ = wrapped[0]  # type: ignore[index]

    # Run the real trainer codepath: this also setattrs `preprocess` /
    # `get_embedding_dimension` onto the wrapper, which would fool a
    # `hasattr(model, "preprocess")` stop condition.
    _StubTrainer().override_model_in_loss(adaptive, wrapped)  # type: ignore[arg-type]
    assert adaptive.model is wrapped
    assert hasattr(wrapped, "preprocess")

    features, labels = _features_and_labels(stsb_bert_tiny_model)
    loss = adaptive(features, labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "wrap",
    [
        lambda m: _FakeCompile(_FakeDDP(m)),
        lambda m: _FakeDDP(_FakeCompile(m)),
    ],
    ids=["compile_of_ddp", "ddp_of_compile"],
)
def test_adaptive_layer_loss_unwraps_nested_wrappers(
    stsb_bert_tiny_model: SentenceTransformer,
    wrap,
) -> None:
    # [FUNCTION DOC]
    # Parametrized over compile-of-DDP and DDP-of-compile nesting; rebinds the loss model to the nested wrapper and asserts a finite scalar loss.
    """Compile-of-DDP and DDP-of-compile both need to unwrap to the inner BaseModel."""
    adaptive = _make_loss(stsb_bert_tiny_model)
    wrapped = wrap(stsb_bert_tiny_model)
    _StubTrainer().override_model_in_loss(adaptive, wrapped)  # type: ignore[arg-type]

    features, labels = _features_and_labels(stsb_bert_tiny_model)
    loss = adaptive(features, labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_adaptive_layer_loss_raises_on_unrecognised_wrapper(stsb_bert_tiny_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Replaces the loss model with a plain linear layer and asserts the forward raises TypeError saying it could not unwrap the model.
    """If `self.model` is something we can't unwrap to a BaseModel, fail loudly instead of
    silently raising the cryptic `'X' object is not subscriptable`."""
    adaptive = _make_loss(stsb_bert_tiny_model)
    adaptive.model = nn.Linear(2, 2)  # neither a BaseModel nor a recognised wrapper

    features, labels = _features_and_labels(stsb_bert_tiny_model)
    with pytest.raises(TypeError, match="could not unwrap"):
        adaptive(features, labels)


def test_adaptive_layer_loss_error_names_inner_stuck_point(stsb_bert_tiny_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Replaces the loss model with the fake DDP wrapping a linear layer and asserts the TypeError names both the outer wrapper class and
    # the inner class where unwrapping stopped.
    """For nested wrappers the error should name both the outer wrapper (what the user
    passed in) and the inner object the unwrap got stuck on."""
    adaptive = _make_loss(stsb_bert_tiny_model)
    adaptive.model = _FakeDDP(nn.Linear(2, 2))  # DDP wraps an unrecognised inner

    features, labels = _features_and_labels(stsb_bert_tiny_model)
    with pytest.raises(TypeError, match=r"could not unwrap _FakeDDP .*stopped at Linear"):
        adaptive(features, labels)


def test_adaptive_layer_loss_kl_teacher_is_detached(stsb_bert_tiny_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # With last-layer and prior-layer weights set to zero, KL weight 1 and temperature 0.3 over all prior layers, runs forward and backward
    # and asserts the last transformer layer's parameters received zero total gradient (the final-layer KL target is detached).
    # Mutates model gradients.
    """Regression test for #3757: the final-layer embeddings are the (teacher) target of the
    self-distillation KL loss and must be detached, so no gradient flows back into the final
    layer through them.

    Isolation: with ``last_layer_weight`` and ``prior_layers_weight`` set to 0 the only loss
    term left is the KL divergence. The *last* transformer layer's parameters influence only
    the teacher (final) embedding, never the student (intermediate) embeddings, so any gradient
    reaching them proves the teacher is still in the autograd graph.
    """
    model = stsb_bert_tiny_model
    inner = MultipleNegativesRankingLoss(model)
    adaptive = AdaptiveLayerLoss(
        model,
        inner,
        n_layers_per_step=-1,  # deterministic: use every prior layer, no random sampling
        last_layer_weight=0.0,
        prior_layers_weight=0.0,
        kl_div_weight=1.0,
        kl_temperature=0.3,
    )
    features, labels = _features_and_labels(model)

    model.zero_grad(set_to_none=True)
    loss = adaptive(features, labels)
    loss.backward()

    last_layer = model[0].auto_model.encoder.layer[-1]
    grad_magnitude = sum(p.grad.abs().sum().item() for p in last_layer.parameters() if p.grad is not None)
    assert grad_magnitude == 0.0, (
        "KL teacher (final_embeddings) must be detached: the final layer received "
        f"gradient magnitude {grad_magnitude} from the KL term."
    )


class _RecordingLoss(nn.Module):
    # [CLASS DOC]
    # HELPER: Loss wrapper that delegates to an inner loss and appends each returned scalar to a values list (final layer first, then prior
    # layers in index order).
    """Wraps an inner loss and records each scalar it returns. AdaptiveLayerLoss calls the
    inner loss once for the final layer, then once per prior layer in layer index order."""

    def __init__(self, inner: nn.Module) -> None:
        # [FUNCTION DOC]
        # HELPER: Stores the inner loss and initializes an empty list of recorded values.
        super().__init__()
        self.inner = inner
        self.values: list[float] = []

    def forward(self, sentence_features, labels):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC]
        # HELPER: Calls the inner loss, records its value as a Python float, and returns the tensor unchanged.
        out = self.inner(sentence_features, labels)
        self.values.append(out.item())
        return out


@pytest.mark.parametrize(
    ("layer_weighting", "weight_fn"),
    [
        ("uniform", lambda layer_idx: 1.0),
        ("linear", lambda layer_idx: 1.0 / (1 + layer_idx)),
        ("log", lambda layer_idx: 1.0 / (1.0 + math.log(1 + layer_idx))),
        (lambda layer_idx: 0.5**layer_idx, lambda layer_idx: 0.5**layer_idx),
    ],
    ids=["uniform", "linear", "log", "callable"],
)
def test_adaptive_layer_loss_layer_weighting(
    stsb_bert_tiny_model: SentenceTransformer,
    layer_weighting,
    weight_fn,
) -> None:
    # [FUNCTION DOC]
    # Parametrized over "uniform", "linear", "log" and a callable weighting, each paired with a reference weight function of the layer index.
    # With KL disabled (temperature 0) and all prior layers used, computes the total loss without gradients and asserts it equals the final
    # layer loss plus the mean over prior layers of weight times loss, and that there are at least two prior layers.
    """Each prior layer loss is scaled by the scheme's weight for its layer index, averaged
    over the number of sampled layers. KL is disabled so the total is exactly that sum."""
    recording = _RecordingLoss(MultipleNegativesRankingLoss(stsb_bert_tiny_model))
    adaptive = AdaptiveLayerLoss(
        stsb_bert_tiny_model,
        recording,
        n_layers_per_step=-1,  # deterministic: use every prior layer in order
        kl_temperature=0.0,
        layer_weighting=layer_weighting,
    )
    features, labels = _features_and_labels(stsb_bert_tiny_model)
    with torch.no_grad():
        total = adaptive(features, labels).item()

    last_layer_loss, *prior_losses = recording.values
    expected = last_layer_loss + sum(
        weight_fn(layer_idx) * value / len(prior_losses) for layer_idx, value in enumerate(prior_losses)
    )
    assert len(prior_losses) >= 2
    assert total == pytest.approx(expected, rel=1e-5)


def test_adaptive_layer_loss_layer_weighting_in_config_dict(stsb_bert_tiny_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts the default config reports layer_weighting "uniform" and that a callable weighting is reported by its function name.
    adaptive = _make_loss(stsb_bert_tiny_model)
    assert adaptive.get_config_dict()["layer_weighting"] == "uniform"

    def halving(layer_idx: int) -> float:
        # [FUNCTION DOC]
        # HELPER: Returns 0.5 raised to the layer index.
        return 0.5**layer_idx

    inner_loss = MultipleNegativesRankingLoss(stsb_bert_tiny_model)
    adaptive = AdaptiveLayerLoss(stsb_bert_tiny_model, inner_loss, layer_weighting=halving)
    assert adaptive.get_config_dict()["layer_weighting"] == "halving"


def test_adaptive_layer_loss_invalid_layer_weighting(stsb_bert_tiny_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Asserts an unknown weighting name raises ValueError mentioning layer_weighting.
    inner_loss = MultipleNegativesRankingLoss(stsb_bert_tiny_model)
    with pytest.raises(ValueError, match="layer_weighting"):
        AdaptiveLayerLoss(stsb_bert_tiny_model, inner_loss, layer_weighting="quadratic")
