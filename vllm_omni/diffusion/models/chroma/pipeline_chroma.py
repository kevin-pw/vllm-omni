"""
Chroma1-HD text-to-image pipeline adapter for vllm-omni.
"""

import torch
import torch.nn as nn
from collections.abc import Iterable

from diffusers import ChromaPipeline as DiffusersChromaPipeline

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import OmniDiffusionRequest


_PIPELINE_COMPONENTS = frozenset(
    {"transformer", "vae", "text_encoder", "tokenizer", "scheduler"}
)


class ChromaPipeline(nn.Module):

    _repeated_blocks = [
        "ChromaTransformerBlock",
        "ChromaSingleTransformerBlock",
    ]

    weights_sources = ()

    def __init__(self, od_config: OmniDiffusionConfig):
        super().__init__()

        model_path = od_config.model

        # Let diffusers handle all config parsing and weight loading.
        # from_pretrained loads weights to CPU by default.
        pipe = DiffusersChromaPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )

        # -------------------------------------------------------
        # KEY FIX: Explicitly move all pipeline components to GPU.
        #
        # from_pretrained loads safetensors to CPU regardless of
        # the outer `with target_device:` context.  The existing
        # FluxPipeline avoids this by manually loading each
        # component with explicit .to(self.device) calls.
        # For our wrapper we move the entire pipeline at once.
        #
        # Skip if CPU offloading is configured — the framework's
        # apply_offload_hooks() will handle device placement later.
        # -------------------------------------------------------
        if not (
            getattr(od_config, "enable_cpu_offload", False)
            or getattr(od_config, "enable_layerwise_offload", False)
        ):
            device = get_local_device()
            pipe = pipe.to(device)

        self._pipe = pipe

    # ------------------------------------------------------------------
    # Transparent attribute delegation
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
    # Weight loading — no-op
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
        prompts_raw = req.prompts or []

        # Extract prompt strings from dicts
        # (same pattern as FluxPipeline.forward)
        prompt = [
            p if isinstance(p, str) else (p.get("prompt") or "")
            for p in prompts_raw
        ]

        # Extract negative prompts
        negative_prompt = None
        if all(
            isinstance(p, str) or p.get("negative_prompt") is None
            for p in prompts_raw
        ):
            negative_prompt = None
        elif prompts_raw:
            negative_prompt = [
                ""
                if isinstance(p, str)
                else (p.get("negative_prompt") or "")
                for p in prompts_raw
            ]

        # Sampling parameters
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

        # Run the diffusers pipeline
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

        return DiffusionOutput(output=output.images)


# ── pre / post processing ────────────────────────────────────────────


def get_chroma_pre_process_func(od_config: OmniDiffusionConfig):
    def pre_process(request):
        return request

    return pre_process


def get_chroma_post_process_func(od_config: OmniDiffusionConfig):
    def post_process(images):
        return images

    return post_process