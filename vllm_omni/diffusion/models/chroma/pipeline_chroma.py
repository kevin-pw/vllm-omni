"""
Chroma1-HD text-to-image pipeline for vllm-omni.
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
    """

    _repeated_blocks = [
        "ChromaTransformerBlock",
        "ChromaSingleTransformerBlock",
    ]

    # -----------------------------------------------------------------
    # THIS IS THE KEY FIX: accept od_config, use it to load the model
    # via diffusers' from_pretrained, then init the parent with the
    # loaded components.
    # -----------------------------------------------------------------
    def __init__(self, od_config: "OmniDiffusionConfig", **kwargs):
        self._od_config = od_config

        # Resolve model path from the omni config
        model_path = od_config.model

        # Let diffusers handle all weight loading, config parsing, etc.
        tmp_pipe = DiffusersChromaPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )

        # Initialise the parent DiffusionPipeline with loaded components
        super().__init__(
            scheduler=tmp_pipe.scheduler,
            text_encoder=tmp_pipe.text_encoder,
            tokenizer=tmp_pipe.tokenizer,
            transformer=tmp_pipe.transformer,
            vae=tmp_pipe.vae,
        )

    # -----------------------------------------------------------------
    # forward() is called by vllm-omni's diffusion executor.
    # Unpack the OmniDiffusionRequest and delegate to the standard
    # diffusers __call__ path.
    # -----------------------------------------------------------------
    def forward(self, request: "OmniDiffusionRequest"):
        generator = None
        seed = getattr(request, "seed", None)
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)

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
    def pre_process(request: "OmniDiffusionRequest"):
        return request
    return pre_process


def get_chroma_post_process_func(od_config: "OmniDiffusionConfig"):
    def post_process(images):
        return images
    return post_process