"""
Chroma1-HD text-to-image pipeline for vllm-omni.

Wraps diffusers' ChromaPipeline inside an nn.Module so that
vllm-omni's model loader (which expects named_parameters(), to(), etc.)
works correctly.  All weights are loaded by diffusers' from_pretrained;
we deliberately expose an empty named_parameters() so that the
loader's load_weights() step becomes a harmless no-op.
"""

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from diffusers import ChromaPipeline as DiffusersChromaPipeline

if TYPE_CHECKING:
    from vllm_omni.diffusion.config import OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest


class ChromaPipeline(nn.Module):
    """
    Chroma text-to-image pipeline adapted for vllm-omni.

    Inherits from nn.Module (required by the framework) and stores the
    actual diffusers pipeline as a plain Python attribute — NOT as a
    registered submodule.  This means:

      • named_parameters() returns empty → load_weights() is a no-op
        (weights are already loaded via from_pretrained).
      • to() / device / dtype are overridden to propagate to the
        wrapped diffusers pipeline.
    """

    _repeated_blocks = [
        "ChromaTransformerBlock",
        "ChromaSingleTransformerBlock",
    ]

    def __init__(self, od_config: "OmniDiffusionConfig"):
        super().__init__()
        self._od_config = od_config
        model_path = od_config.model

        # ----------------------------------------------------------
        # Load via diffusers — this fetches configs, instantiates
        # T5, ChromaTransformer2DModel, VAE, scheduler, and loads
        # ALL weights from the safetensors.
        # ----------------------------------------------------------
        pipe = DiffusersChromaPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )

        # Store as a plain attribute.  nn.Module.__setattr__ only
        # auto-registers values that are nn.Module instances;
        # DiffusersChromaPipeline is NOT an nn.Module (it inherits
        # from ConfigMixin), so this goes through the regular
        # object.__setattr__ path.  Result: named_parameters()
        # stays empty, which is exactly what we need.
        self._pipe = pipe

    # ------------------------------------------------------------------
    # Device / dtype handling
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):
        """Propagate device / dtype moves to the diffusers pipeline."""
        # DiffusionPipeline.to() accepts the same positional forms
        # as nn.Module.to() in recent diffusers (device, dtype, etc.)
        self._pipe.to(*args, **kwargs)
        return self

    @property
    def device(self):
        return self._pipe.device

    # ------------------------------------------------------------------
    # forward() — called by vllm-omni's diffusion executor
    # ------------------------------------------------------------------
    def forward(self, request: "OmniDiffusionRequest"):
        generator = None
        seed = getattr(request, "seed", None)
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)

        output = self._pipe(
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", None),
            num_inference_steps=getattr(
                request, "num_inference_steps", None
            )
            or 40,
            guidance_scale=getattr(request, "guidance_scale", None) or 3.0,
            height=getattr(request, "height", None),
            width=getattr(request, "width", None),
            num_images_per_prompt=getattr(
                request, "num_images_per_prompt", 1
            ),
            generator=generator,
        )
        return output.images


# ======================================================================
# Pre / post-processing factories (pass-through for pure T2I)
# ======================================================================


def get_chroma_pre_process_func(od_config: "OmniDiffusionConfig"):
    def pre_process(request: "OmniDiffusionRequest"):
        return request

    return pre_process


def get_chroma_post_process_func(od_config: "OmniDiffusionConfig"):
    def post_process(images):
        return images

    return post_process