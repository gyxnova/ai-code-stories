import os
from pathlib import Path

import torch
import spaces
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


NEGATIVE_PROMPT = (
    "lowres, bad anatomy, worst quality, text, watermark, blurry, extra limbs, "
    "goggles, denim, 1girl, female, feminine, bare legs"
)

_pipe = None

# Project root.
# In Hugging Face Spaces this should normally be /app.
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

NARRATOR_LORA = MODELS_DIR / "narrator_lora.safetensors"
WORKER_LORA = MODELS_DIR / "worker.safetensors"


def get_pipeline():
    """Load the base model + both LoRAs once and reuse them."""

    global _pipe

    if _pipe is not None:
        return _pipe

    # ---------------------------------------------------------
    # DEBUG: verify the LoRA files exist
    # ---------------------------------------------------------
    print("========================================")
    print("Image Generator Startup")
    print("BASE_DIR:", BASE_DIR)
    print("MODELS_DIR:", MODELS_DIR)
    print("MODELS_DIR EXISTS:", MODELS_DIR.exists())
    print("NARRATOR_LORA:", NARRATOR_LORA)
    print("NARRATOR_LORA EXISTS:", NARRATOR_LORA.exists())
    print("WORKER_LORA:", WORKER_LORA)
    print("WORKER_LORA EXISTS:", WORKER_LORA.exists())

    if MODELS_DIR.exists():
        print("FILES IN MODELS:", os.listdir(MODELS_DIR))
    else:
        print("ERROR: models directory does not exist!")

    print("========================================")

    # ---------------------------------------------------------
    # Make sure the LoRA files are actually present
    # ---------------------------------------------------------
    if not NARRATOR_LORA.exists():
        raise FileNotFoundError(
            f"Narrator LoRA not found at: {NARRATOR_LORA}"
        )

    if not WORKER_LORA.exists():
        raise FileNotFoundError(
            f"Worker LoRA not found at: {WORKER_LORA}"
        )

    # ---------------------------------------------------------
    # Download base Animagine XL model from Hugging Face
    # ---------------------------------------------------------
    checkpoint_path = hf_hub_download(
        repo_id="cagliostrolab/animagine-xl-4.0",
        filename="animagine-xl-4.0.safetensors",
    )

    print("Base checkpoint:", checkpoint_path)

    # ---------------------------------------------------------
    # Load Stable Diffusion XL pipeline
    # ---------------------------------------------------------
    pipe = StableDiffusionXLPipeline.from_single_file(
        checkpoint_path,
        torch_dtype=torch.float16,
    )

    # ---------------------------------------------------------
    # Load LoRA files explicitly as LOCAL safetensors
    # ---------------------------------------------------------
    print("Loading narrator LoRA...")
    narrator_state_dict = load_file(str(NARRATOR_LORA))

    print("Loading worker LoRA...")
    worker_state_dict = load_file(str(WORKER_LORA))

    # Load narrator LoRA
    pipe.load_lora_weights(
        narrator_state_dict,
        adapter_name="narrator",
    )

    # Load worker LoRA
    pipe.load_lora_weights(
        worker_state_dict,
        adapter_name="worker",
    )

    print("Both LoRAs loaded successfully.")

    # ---------------------------------------------------------
    # VAE memory optimizations
    # ---------------------------------------------------------
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()

    _pipe = pipe

    return pipe


def build_prompt(panel: dict) -> str:
    narrator_desc = (
        "narrator, 1boy, male, long messy pink hair, blue eyes, "
        "dangling earring, oversized white t-shirt, black cargo pants, "
        "white sneakers"
    )

    worker_descs = [
        f"chibi robot worker, {color} glowing eyes, {color} chest stripe"
        for color in panel.get("workers_present", [])
    ]

    parts = [
        panel["narrator_action"],
        narrator_desc,
        *worker_descs,
        panel["scene"],
        "cel shading, anime coloring, masterpiece",
    ]

    return ", ".join(parts)


@spaces.GPU
def generate_panel(panel: dict, seed: int = None):

    # ---------------------------------------------------------
    # Get pipeline and move it to GPU
    # ---------------------------------------------------------
    pipe = get_pipeline().to("cuda")

    # ---------------------------------------------------------
    # Build prompt
    # ---------------------------------------------------------
    prompt = build_prompt(panel)

    # ---------------------------------------------------------
    # Select LoRAs
    # ---------------------------------------------------------
    active_adapters = ["narrator"]
    weights = [1.0]

    if panel.get("workers_present"):
        active_adapters.append("worker")
        weights.append(1.0)

    pipe.set_adapters(
        active_adapters,
        adapter_weights=weights,
    )

    # ---------------------------------------------------------
    # Random generator
    # ---------------------------------------------------------
    generator = torch.Generator(device="cpu")

    if seed is not None:
        generator = generator.manual_seed(seed)

    # ---------------------------------------------------------
    # Generate image
    # ---------------------------------------------------------
    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=25,
        guidance_scale=7,
        generator=generator,
    ).images[0]

    return image