from typing import Dict

import pytest
import torch

from flux_regional_prompting.models.attention_processor import (
    RegionalFluxAttnProcessor2_0,
)
from flux_regional_prompting.models.transformers import RegionalFluxTransformer2DModel
from flux_regional_prompting.pipelines import (
    RegionalFluxPipeline,
)
from flux_regional_prompting.utils import RegionalPromptMask


@pytest.fixture
def model_id() -> str:
    return "black-forest-labs/FLUX.1-dev"


@pytest.fixture
def torch_dtype() -> torch.dtype:
    return torch.bfloat16


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def image_width() -> int:
    return 1280


@pytest.fixture
def image_height() -> int:
    return 768


@pytest.fixture
def num_inference_steps() -> int:
    return 24


@pytest.fixture
def guidance_scale() -> float:
    return 3.5


@pytest.fixture
def seed() -> int:
    return 124


@pytest.fixture
def base_prompt() -> str:
    return "An ancient woman stands solemnly holding a blazing torch, while a fierce battle rages in the background, capturing both strength and tragedy in a historical war scene."


@pytest.fixture
def background_prompt() -> str:
    return "a photo"


@pytest.fixture
def regional_prompt_mask_pairs() -> Dict[str, RegionalPromptMask]:
    return {
        "0": {
            "description": "A dignified woman in ancient robes stands in the foreground, her face illuminated by the torch she holds high. Her expression is one of determination and sorrow, her clothing and appearance reflecting the historical period. The torch casts dramatic shadows across her features, its flames dancing vibrantly against the darkness",
            "mask": (128, 128, 640, 768),
        }
    }


@pytest.fixture
def mask_inject_steps() -> int:
    return 10  # larger means stronger control, recommended between 5-10


@pytest.fixture
def double_inject_blocks_interval() -> int:
    return 1  # 1 means strongest control


@pytest.fixture
def single_inject_blocks_interval() -> int:
    return 1  # 1 means strongest control


@pytest.fixture
def base_ratio() -> float:
    return 0.3  # smaller means stronger control


def test_pipeline_flux(
    model_id: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    #
    # General settings
    #
    image_width: int,
    image_height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    base_prompt: str,
    background_prompt: str,
    regional_prompt_mask_pairs: Dict[str, RegionalPromptMask],
    #
    # Region control factor settings
    #
    mask_inject_steps: int,
    double_inject_blocks_interval: int,
    single_inject_blocks_interval: int,
    base_ratio: float,
):
    transformer = RegionalFluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
    )
    transformer = transformer.to(torch_dtype)

    pipe = RegionalFluxPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)

    attn_procs = {}
    for name in pipe.transformer.attn_processors.keys():
        if "transformer_blocks" in name and name.endswith("attn.processor"):
            attn_procs[name] = RegionalFluxAttnProcessor2_0()
        else:
            attn_procs[name] = pipe.transformer.attn_processors[name]
    pipe.transformer.set_attn_processor(attn_procs)

    regional_prompts = []
    regional_masks = []
    background_mask = torch.ones((image_height, image_width))
    for region_idx, region in regional_prompt_mask_pairs.items():
        description = region["description"]
        mask = region["mask"]
        x1, y1, x2, y2 = mask
        mask = torch.zeros((image_height, image_width))
        mask[y1:y2, x1:x2] = 1.0
        background_mask -= mask
        regional_prompts.append(description)
        regional_masks.append(mask)

    # if regional masks don't cover the whole image, append background prompt and mask
    if background_mask.sum() > 0:
        regional_prompts.append(background_prompt)
        regional_masks.append(background_mask)

    image = pipe(
        prompt=base_prompt,
        width=image_width,
        height=image_height,
        mask_inject_steps=mask_inject_steps,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
        joint_attention_kwargs={
            "regional_prompts": regional_prompts,
            "regional_masks": regional_masks,
            "double_inject_blocks_interval": double_inject_blocks_interval,
            "single_inject_blocks_interval": single_inject_blocks_interval,
            "base_ratio": base_ratio,
        },
    ).images[0]

    image.save("output.png")
