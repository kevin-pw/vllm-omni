"""
Chroma1-HD text-to-image pipeline adapter for vllm-omni.

Wraps diffusers' ChromaPipeline inside an nn.Module, transparently
delegating attribute access so that the framework can reach .vae,
.transformer, .scheduler, .device, etc.
"""

import warnings
from collections.abc import Iterable

# Suppress known deprecation warnings from diffusers internals
warnings.filterwarnings(
    "ignore", message=".*FluxPosEmbed.*is deprecated.*"
)
warnings.filterwarnings(
    "ignore", message=".*torch_dtype.*is deprecated.*Use.*dtype.*instead.*"
)

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from diffusers import ChromaPipeline as DiffusersChromaPipeline

if TYPE_CHECKING:
    from vllm_omni.diffusion.config import OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest


# Names of pipeline components that the framework may assign to
# (e.g. `model.transformer = regionally_compile(model.transformer)`)
_PIPELINE_COMPONENTS = frozenset(
    {"transformer", "vae", "text_encoder", "tokenizer", "scheduler"}
)


class ChromaPipeline(nn.Module):
    """
    nn.Module wrapper around diffusers' ChromaPipeline.

    Key design decisions
    --------------------
    * The real diffusers pipeline is stored as ``self._pipe``, a plain
      ``__dict__`` entry (DiffusersChromaPipeline is NOT nn.Module, so
      nn.Module.__setattr__ does not register it as a submodule).
    * ``named_parameters()`` is therefore empty → ``load_weights()`` is
      a harmless no-op (all weights were already loaded by
      ``from_pretrained``).
    * ``__getattr__`` falls back to ``self._pipe``, exposing ``.vae``,
      ``.transformer``, ``.scheduler``, ``.device``, … to the framework.
    * ``__setattr__`` intercepts component assignments (e.g. after
      ``torch.compile``) and forwards them to ``self._pipe``.
    """

    # Block class names used by torch.compile regional compilation
    _repeated_blocks = [
        "ChromaTransformerBlock",
        "ChromaSingleTransformerBlock",
    ]

    # No external weight files — from_pretrained loaded everything
    weights_sources = ()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, od_config: "OmniDiffusionConfig"):
        super().__init__()

        model_path = od_config.model

        # Detect which device the framework wants (set via the
        # ``with target_device:`` context manager in diffusers_loader).
        target = torch.empty(0).device

        # Load on CPU first, regardless of the outer device context,
        # to avoid blowing GPU memory with intermediate allocations.
        with torch.device("cpu"):
            pipe = DiffusersChromaPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )

        # Move to the target device when it is a GPU
        if target.type != "cpu":
            pipe = pipe.to(target)

        # Stored as a plain dict attribute — NOT a registered submodule
        self._pipe = pipe

    # ------------------------------------------------------------------
    # Transparent attribute delegation
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        """Fall back to the wrapped diffusers pipeline for any attribute
        the nn.Module itself does not have (.vae, .transformer,
        .scheduler, .text_encoder, .device, …)."""
        # Guard: during __init__, _pipe may not exist yet
        _pipe = self.__dict__.get("_pipe")
        if _pipe is not None:
            try:
                return getattr(_pipe, name)
            except AttributeError:
                pass
        # Default nn.Module behaviour (checks _modules, _parameters, …)
        return super().__getattr__(name)

    def __setattr__(self, name: str, value):
        """Forward component assignments to the wrapped pipeline so that
        e.g. ``model.transformer = compiled_transformer`` propagates."""
        if name in _PIPELINE_COMPONENTS and "_pipe" in self.__dict__:
            setattr(self._pipe, name, value)
            return
        super().__setattr__(name, value)

    # ------------------------------------------------------------------
    # Device / dtype helpers
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):
        self._pipe.to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Weight loading — no-op
    # ------------------------------------------------------------------
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ):
        """All weights were loaded by from_pretrained.
        Consume the (empty) iterator and return None to skip the
        loaded-weights check."""
        for _ in weights:
            pass
        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def forward(self, request: "OmniDiffusionRequest"):
        sp = request.sampling_params

        output = self._pipe(
            prompt=request.prompts,
            negative_prompt=(
                getattr(request, "negative_prompt", None)
                or getattr(request, "negative_prompts", None)
            ),
            num_inference_steps=(
                getattr(sp, "num_inference_steps", None) or 40
            ),
            guidance_scale=(
                getattr(sp, "guidance_scale", None) or 3.0
            ),
            height=(
                getattr(sp, "height", None)
                or getattr(request, "height", None)
            ),
            width=(
                getattr(sp, "width", None)
                or getattr(request, "width", None)
            ),
            num_images_per_prompt=getattr(
                sp, "num_images_per_prompt", 1
            ),
            generator=getattr(sp, "generator", None),
        )
        return output.images


# ── pre / post processing (pass-through for pure T2I) ────────────────


def get_chroma_pre_process_func(od_config: "OmniDiffusionConfig"):
    def pre_process(request):
        return request

    return pre_process


def get_chroma_post_process_func(od_config: "OmniDiffusionConfig"):
    def post_process(images):
        return images

    return post_process