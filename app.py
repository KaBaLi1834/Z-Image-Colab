
#@title Utils Code

import os, random, time, shutil
import torch
import numpy as np
from PIL import Image
import re, uuid
from nodes import NODE_CLASS_MAPPINGS

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()

# PuLID nodes
PulidFluxInsightFaceLoader = NODE_CLASS_MAPPINGS["PulidFluxInsightFaceLoader"]()
PulidFluxEvaClipLoader = NODE_CLASS_MAPPINGS["PulidFluxEvaClipLoader"]()
PulidFluxModelLoader = NODE_CLASS_MAPPINGS["PulidFluxModelLoader"]()
ApplyPulidFlux = NODE_CLASS_MAPPINGS["ApplyPulidFlux"]()

with torch.inference_mode():
    unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
    clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
    vae = VAELoader.load_vae("ae.safetensors")[0]
    # Load PuLID models once at startup
    insightface = PulidFluxInsightFaceLoader.load_insightface("CPU")[0]
    eva_clip = PulidFluxEvaClipLoader.load_eva_clip()[0]
    pulid_model = PulidFluxModelLoader.load_model("pulid_flux_v0.9.1.safetensors")[0]

LORA_DIR = "./models/loras"
os.makedirs(LORA_DIR, exist_ok=True)

save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)


def get_save_path(prompt):
    safe_prompt = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)[:25]
    uid = uuid.uuid4().hex[:6]
    filename = f"{safe_prompt}_{uid}.png"
    return os.path.join(save_dir, filename)


def upload_lora(lora_file):
    if lora_file is None:
        return gr.update(choices=get_lora_list(), value="None"), "No file uploaded."
    dest = os.path.join(LORA_DIR, os.path.basename(lora_file))
    shutil.copy(lora_file, dest)
    choices = get_lora_list()
    fname = os.path.basename(lora_file)
    return gr.update(choices=choices, value=fname), f"✅ LoRA '{fname}' uploaded successfully!"


def get_lora_list():
    files = ["None"] + [f for f in os.listdir(LORA_DIR) if f.endswith((".safetensors", ".pt", ".ckpt"))]
    return files


def pil_to_comfy_tensor(pil_img):
    """Convert a PIL image to the float tensor format ComfyUI expects: [1, H, W, 3] 0-1."""
    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, 3]


@torch.inference_mode()
def generate(input):
    values = input["input"]
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    seed = values['seed']
    steps = values['steps']
    cfg = values['cfg']
    sampler_name = values['sampler_name']
    scheduler = values['scheduler']
    denoise = values['denoise']
    width = values['width']
    height = values['height']
    batch_size = values['batch_size']
    lora_name = values.get('lora_name', 'None')
    lora_strength = values.get('lora_strength', 1.0)
    face_image = values.get('face_image', None)        # PIL image or None
    pulid_weight = values.get('pulid_weight', 0.9)
    pulid_start = values.get('pulid_start', 0.0)
    pulid_end = values.get('pulid_end', 1.0)

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)

    active_unet = unet
    active_clip = clip

    # Apply LoRA
    if lora_name and lora_name != "None":
        active_unet, active_clip = LoraLoader.load_lora(
            active_unet, active_clip, lora_name,
            strength_model=lora_strength,
            strength_clip=lora_strength
        )

    # Apply PuLID FaceID
    if face_image is not None:
        face_tensor = pil_to_comfy_tensor(face_image)
        active_unet = ApplyPulidFlux.apply_pulid(
            active_unet, pulid_model, eva_clip, insightface,
            face_tensor,
            weight=pulid_weight,
            start_at=pulid_start,
            end_at=pulid_end
        )[0]

    positive = CLIPTextEncode.encode(active_clip, positive_prompt)[0]
    negative = CLIPTextEncode.encode(active_clip, negative_prompt)[0]
    latent_image = EmptyLatentImage.generate(width, height, batch_size=batch_size)[0]
    samples = KSampler.sample(active_unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
    decoded = VAEDecode.decode(vae, samples)[0].detach()
    save_path = get_save_path(positive_prompt)
    Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0]).save(save_path)
    return save_path, seed


import gradio as gr


def generate_ui(
    positive_prompt,
    negative_prompt,
    aspect_ratio,
    seed,
    steps,
    cfg,
    denoise,
    lora_name,
    lora_strength,
    face_image,
    pulid_weight,
    pulid_start,
    pulid_end,
    batch_size=1,
    sampler_name="euler",
    scheduler="simple"
):
    width, height = [int(x) for x in aspect_ratio.split("(")[0].strip().split("x")]

    input_data = {
        "input": {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "batch_size": int(batch_size),
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": float(denoise),
            "lora_name": lora_name,
            "lora_strength": float(lora_strength),
            "face_image": face_image,      # PIL image from gr.Image
            "pulid_weight": float(pulid_weight),
            "pulid_start": float(pulid_start),
            "pulid_end": float(pulid_end),
        }
    }

    image_path, seed = generate(input_data)
    return image_path, image_path, seed


DEFAULT_POSITIVE = """A beautiful woman with platinum blond hair that is almost white, snowy white skin, red bush, very big plump red lips, high cheek bones and sharp. She has almond shaped red eyes and she's holding a intricate mask. She's wearing white and gold royal gown with a black cloak.  In the veins of her neck its gold."""

DEFAULT_NEGATIVE = """low quality, blurry, unnatural skin tone, bad lighting, pixelated,
noise, oversharpen, soft focus, pixelated"""

ASPECTS = [
    "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1152x864 (4:3)", "864x1152 (3:4)", "1248x832 (3:2)",
    "832x1248 (2:3)", "1280x720 (16:9)", "720x1280 (9:16)",
    "1344x576 (21:9)", "576x1344 (9:21)"
]

custom_css = ".gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
<div style="width:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; margin:20px 0;">
    <h1 style="font-size:2.5em; margin-bottom:10px;">Z-Image-Turbo</h1>
    <a href="https://github.com/Tongyi-MAI/Z-Image" target="_blank">
        <img src="https://img.shields.io/badge/GitHub-Z--Image-181717?logo=github&logoColor=white"
             style="height:15px;">
    </a>
</div>
""")

    with gr.Row():
        with gr.Column():
            positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=5)

            with gr.Row():
                aspect = gr.Dropdown(ASPECTS, value="1024x1024 (1:1)", label="Aspect Ratio")
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                steps = gr.Slider(4, 25, value=9, step=1, label="Steps")

            with gr.Row():
                run = gr.Button('🚀 Generate', variant='primary')

            with gr.Accordion('Image Settings', open=False):
                with gr.Row():
                    cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                    denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")
                with gr.Row():
                    negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=3)

            # ── LoRA Section ──────────────────────────────────────────────
            with gr.Accordion('🎭 Custom LoRA (Consistent Character)', open=False):
                lora_upload = gr.File(
                    label="Upload LoRA (.safetensors / .pt / .ckpt)",
                    file_types=[".safetensors", ".pt", ".ckpt"],
                    type="filepath"
                )
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                lora_select = gr.Dropdown(
                    choices=get_lora_list(),
                    value="None",
                    label="Select LoRA"
                )
                lora_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA Strength")

                lora_upload.upload(
                    fn=upload_lora,
                    inputs=[lora_upload],
                    outputs=[lora_select, upload_status]
                )

            # ── PuLID FaceID Section ──────────────────────────────────────
            with gr.Accordion('🧬 PuLID FaceID (Consistent Face)', open=False):
                gr.Markdown("Upload a **clear, front-facing** reference photo. Leave empty to disable.")
                face_image = gr.Image(label="Reference Face Image", type="pil", sources=["upload"])
                with gr.Row():
                    pulid_weight = gr.Slider(0.0, 1.5, value=0.9, step=0.05, label="FaceID Weight")
                    pulid_start = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Start At")
                    pulid_end = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="End At")
            # ─────────────────────────────────────────────────────────────

        with gr.Column():
            download_image = gr.File(label="Download Image")
            output_img = gr.Image(label="Generated Image", height=480)
            used_seed = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)

    run.click(
        fn=generate_ui,
        inputs=[
            positive, negative, aspect, seed, steps, cfg, denoise,
            lora_select, lora_strength,
            face_image, pulid_weight, pulid_start, pulid_end
        ],
        outputs=[download_image, output_img, used_seed]
    )

demo.launch(share=True, debug=True)
