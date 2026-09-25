import torch
import spaces
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


NEGATIVE_PROMPT = (
    "lowres, bad anatomy, worst quality, text, watermark, blurry, extra limbs, "
    "goggles, denim, 1girl, female, feminine, bare legs, "
    "multiple people, duplicate, twins, clone, extra person, two men, "
    "child, chibi body, small body, kid, young child"
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
    narrator_lora_path = hf_hub_download(
        repo_id="gyxnova/ai-code-stories-loras",
        filename="narrator_lora.safetensors",
    )
    worker_lora_path = hf_hub_download(
        repo_id="gyxnova/ai-code-stories-loras",
        filename="workers.safetensors",
    )

    pipe.load_lora_weights(narrator_lora_path, adapter_name="narrator")
    pipe.load_lora_weights(worker_lora_path, adapter_name="worker")
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()

    _pipe = pipe
    return pipe


def build_prompt(panel: dict) -> str:
    narrator_desc = (
        "solo, 1boy, male, adult, narrator, long messy pink hair, blue eyes, "
        "dangling earring, oversized white t-shirt, black cargo pants, "
        "white sneakers"
    )
    worker_descs = [
        f"(chibi robot worker:1.3), ({color} glowing eyes:1.3), {color} chest stripe"
        for color in panel.get("workers_present", [])
    ]
    parts = [panel["narrator_action"], narrator_desc, *worker_descs,
              panel["scene"], "cel shading, anime coloring, masterpiece"]
    return ", ".join(parts)




@spaces.GPU
def debug_three_way_test(panel: dict, seed: int = 42):
    """Diagnostic: generates narrator-only, worker-only, and both-together
    versions of the same panel, to isolate whether LoRA combination is the
    actual problem."""
    pipe = get_pipeline().to("cuda")

    prompt = build_prompt(panel)

    pipe.set_adapters(["narrator"], adapter_weights=[1.0])
    img_narrator_only = pipe(
        prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=25, guidance_scale=7,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]

    pipe.set_adapters(["worker"], adapter_weights=[1.0])
    img_worker_only = pipe(
        prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=25, guidance_scale=7,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]

    pipe.set_adapters(["narrator", "worker"], adapter_weights=[1.0, 0.5])
    img_both = pipe(
        prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=25, guidance_scale=7,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]

    return [img_narrator_only, img_worker_only, img_both]