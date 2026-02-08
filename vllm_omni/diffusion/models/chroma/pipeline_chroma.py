"""
Chroma1-HD text-to-image pipeline adapter for vllm-omni.

Wraps diffusers' ChromaPipeline inside an nn.Module-compatible shell,
translating OmniDiffusionRequest prompt dicts into the plain strings
that diffusers expects.
"""

import torch
import torch.nn as nn
from collections.abc import Iterable

from diffusers import ChromaPipeline as DiffusersChromaPipeline

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest


# Attribute names that should be read from / written to the wrapped
# diffusers pipeline rather than the nn.Module wrapper itself.
_PIPELINE_COMPONENTS = frozenset(
    {"transformer", "vae", "text_encoder", "tokenizer", "scheduler"}
)


class ChromaPipeline(nn.Module):
    """nn.Module wrapper around diffusers' ChromaPipeline."""

    _repeated_blocks = [
        "ChromaTransformerBlock",
        "ChromaSingleTransformerBlock",
    ]

    # No external weight sources – from_pretrained loaded everything.
    weights_sources = ()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, od_config: OmniDiffusionConfig):
        super().__init__()

        model_path = od_config.model

        # Let diffusers handle all weight loading and config parsing.
        pipe = DiffusersChromaPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )

        # Store as a plain dict attribute – NOT a registered submodule.
        # (DiffusersChromaPipeline is not an nn.Module.)
        self._pipe = pipe

    # ------------------------------------------------------------------
    # Transparent attribute delegation so that the framework can
    # access .vae, .transformer, .scheduler, .device, etc.
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        _pipe = self.__dict__.get("_pipe")
        if _pipe is not None:
            try:
                return getattr(_pipe, name)
            except AttributeError:
                pass
        return super().__getattr__(name)

    def __setattr__(self, name: str, value):
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
    # Weight loading – no-op (already loaded by from_pretrained)
    # ------------------------------------------------------------------
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ):
        for _ in weights:
            pass
        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def forward(self, req: OmniDiffusionRequest):
        """Extract prompts from the OmniDiffusionRequest and run the
        diffusers ChromaPipeline.

        req.prompts is a list where each element is either:
          - str:  a plain prompt string
          - dict: {"prompt": "...", "negative_prompt": "...", ...}

        This follows the same extraction pattern used by the existing
        FluxPipeline.forward() in pipeline_flux.py.
        """

        # ----- prompt extraction (same pattern as FluxPipeline) ------
        prompts_raw = req.prompts or []

        prompt = [
            p if isinstance(p, str) else (p.get("prompt") or "")
            for p in prompts_raw
        ]

        # Extract negative prompts only if at least one exists
        negative_prompt = None
        if all(
            isinstance(p, str) or p.get("negative_prompt") is None
            for p in prompts_raw
        ):
            negative_prompt = None
        elif prompts_raw:
            negative_prompt = [
                "" if isinstance(p, str)
                else (p.get("negative_prompt") or "")
                for p in prompts_raw
            ]

        # ----- sampling parameters -----------------------------------
        sp = req.sampling_params

        height = getattr(sp, "height", None) or 1024
        width = getattr(sp, "width", None) or 1024
        num_inference_steps = (
            getattr(sp, "num_inference_steps", None) or 40
        )
        guidance_scale = (
            sp.guidance_scale
            if getattr(sp, "guidance_scale", None) is not None
            else 3.0
        )
        generator = getattr(sp, "generator", None)
        num_images_per_prompt = max(
            getattr(sp, "num_outputs_per_prompt", 0), 1
        )

        # ----- run diffusers pipeline --------------------------------
        output = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
        )

        # Return DiffusionOutput.  The diffusers pipeline already
        # returns fully post-processed PIL images, so our
        # get_chroma_post_process_func is a pass-through.
        return DiffusionOutput(output=output.images)


# ── pre / post processing ────────────────────────────────────────────


def get_chroma_pre_process_func(od_config: OmniDiffusionConfig):
    """No pre-processing needed for pure text-to-image."""

    def pre_process(request):
        return request

    return pre_process


def get_chroma_post_process_func(od_config: OmniDiffusionConfig):
    """The diffusers pipeline already returns PIL images,
    so post-processing is a pass-through."""

    def post_process(images):
        return images

    return post_process