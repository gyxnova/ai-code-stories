import torch
import spaces
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

NEGATIVE_PROMPT = (
    "lowres, bad anatomy, worst quality, text, watermark, blurry, extra limbs, "
    "goggles, denim, 1girl, female, feminine, bare legs"
)

_pipe = None


def get_pipeline():
    """Load the base model + both LoRAs once, reused across all panel generations.
    Runs on Space startup — base checkpoint is downloaded from HF Hub and cached,
    never committed to the repo."""
    global _pipe
    if _pipe is not None:
        return _pipe

    checkpoint_path = hf_hub_download(
        repo_id="cagliostrolab/animagine-xl-4.0",
        filename="animagine-xl-4.0.safetensors",
    )

    pipe = StableDiffusionXLPipeline.from_single_file(
        checkpoint_path,
        torch_dtype=torch.float16,
    )
    pipe.load_lora_weights("narrator_lora.safetensors", adapter_name="narrator")
    pipe.load_lora_weights("worker_lora.safetensors", adapter_name="worker")
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()

    _pipe = pipe
    return pipe


def build_prompt(panel: dict) -> str:
    narrator_desc = (
        "narrator, 1boy, male, long messy pink hair, blue eyes, dangling earring, "
        "oversized white t-shirt, black cargo pants, white sneakers"
    )
    worker_descs = [
        f"chibi robot worker, {color} glowing eyes, {color} chest stripe"
        for color in panel.get("workers_present", [])
    ]
    parts = [panel["narrator_action"], narrator_desc, *worker_descs,
              panel["scene"], "cel shading, anime coloring, masterpiece"]
    return ", ".join(parts)


@spaces.GPU  # ZeroGPU: this decorator is what actually gets you GPU time on the Space
def generate_panel(panel: dict, seed: int = None):
    pipe = get_pipeline().to("cuda")  # moved to cuda only inside the GPU-decorated call

    prompt = build_prompt(panel)
    active_adapters = ["narrator"]
    weights = [1.0]
    if panel.get("workers_present"):
        active_adapters.append("worker")
        weights.append(1.0)
    pipe.set_adapters(active_adapters, adapter_weights=weights)

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator = generator.manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=25,
        guidance_scale=7,
        generator=generator,
    ).images[0]

    return image