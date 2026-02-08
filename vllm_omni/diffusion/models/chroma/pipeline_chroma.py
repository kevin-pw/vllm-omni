"""
Chroma1-HD text-to-image pipeline for vllm-omni.

Minimal adapter that wraps diffusers' ChromaPipeline to work within
the vllm-omni diffusion framework.  All heavy lifting (text encoding,
denoising loop, VAE decode) is delegated to diffusers.
"""

import torch
from typing import TYPE_CHECKING

from diffusers import ChromaPipeline as DiffusersChromaPipeline

if TYPE_CHECKING:
    from vllm_omni.diffusion.config import OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest


class ChromaPipeline(DiffusersChromaPipeline):
    """
    Chroma text-to-image pipeline adapted for vllm-omni.

    Inherits from diffusers' ChromaPipeline so that ``from_pretrained``
    and all component loading (T5, ChromaTransformer2DModel, VAE,
    scheduler) work out of the box.
    """

    # Block class names inside ChromaTransformer2DModel used by
    # torch.compile to identify the repeated computation.
    # Verify against your diffusers version; if the names changed,
    # update them here – the model will still run, just without the
    # compile speed-up.
    _repeated_blocks = [
        "ChromaTransformerBlock",
        "ChromaSingleTransformerBlock",
    ]

    # ------------------------------------------------------------------
    # forward() is the entry-point that vllm-omni's diffusion executor
    # calls.  We unpack the OmniDiffusionRequest and delegate to the
    # parent's __call__ (the standard diffusers inference path).
    # ------------------------------------------------------------------
    def forward(self, request: "OmniDiffusionRequest"):
        # --- seed / generator ----------------------------------------
        generator = None
        seed = getattr(request, "seed", None)
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)

        # --- call diffusers' full pipeline ----------------------------
        output = DiffusersChromaPipeline.__call__(
            self,
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", None),
            num_inference_steps=getattr(request, "num_inference_steps", None) or 40,
            guidance_scale=getattr(request, "guidance_scale", None) or 3.0,
            height=getattr(request, "height", None),
            width=getattr(request, "width", None),
            num_images_per_prompt=getattr(request, "num_images_per_prompt", 1),
            generator=generator,
        )
        return output.images


# ======================================================================
# Pre / post-processing factories
# ======================================================================

def get_chroma_pre_process_func(od_config: "OmniDiffusionConfig"):
    """
    Return the pre-processing callable for Chroma.

    Chroma is a pure text-to-image pipeline, so there are no input
    images to resize or encode.  Pre-processing is a pass-through.
    """

    def pre_process(request: "OmniDiffusionRequest"):
        return request

    return pre_process


def get_chroma_post_process_func(od_config: "OmniDiffusionConfig"):
    """
    Return the post-processing callable for Chroma.

    Diffusers' ChromaPipeline already returns PIL images, so
    post-processing is a pass-through.
    """

    def post_process(images):
        return images

    return post_process