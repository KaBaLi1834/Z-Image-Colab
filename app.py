#@title Write new app.py with face-lock built in
app_code = '''
import os, random, time, shutil, re, uuid
import torch
import numpy as np
from PIL import Image
from nodes import NODE_CLASS_MAPPINGS

# ── Load base models ──
UNETLoader   = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader   = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader    = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode  = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler        = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode       = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
LoraLoader      = NODE_CLASS_MAPPINGS["LoraLoader"]()

with torch.inference_mode():
    unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
    clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
    vae  = VAELoader.load_vae("ae.safetensors")[0]

# ── ReActor face swap (load once) ──
try:
    ReActorFaceSwap = NODE_CLASS_MAPPINGS["ReActorFaceSwap"]()
    REACTOR_AVAILABLE = True
    print("✅ ReActor face swap loaded")
except Exception as e:
    REACTOR_AVAILABLE = False
    print(f"⚠️ ReActor not available: {e}")

LORA_DIR = "./models/loras"
FACE_REF_DIR = "./input/face_refs"
os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(FACE_REF_DIR, exist_ok=True)
os.makedirs("./results", exist_ok=True)

def get_save_path(prompt):
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", prompt)[:25]
    return f"./results/{safe}_{uuid.uuid4().hex[:6]}.png"

def get_lora_choices():
    files = [f for f in os.listdir(LORA_DIR)
             if f.endswith((".safetensors", ".pt", ".ckpt"))]
    return ["None"] + files

def get_face_ref_choices():
    files = [f for f in os.listdir(FACE_REF_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    return ["None"] + files

def upload_lora(lora_file):
    if lora_file is None:
        return "No file uploaded.", get_lora_choices()
    dest = os.path.join(LORA_DIR, os.path.basename(lora_file))
    shutil.copy(lora_file, dest)
    return f"✅ LoRA uploaded: {os.path.basename(dest)}", get_lora_choices()

def upload_face_ref(face_file):
    if face_file is None:
        return "No file uploaded.", get_face_ref_choices()
    dest = os.path.join(FACE_REF_DIR, os.path.basename(face_file))
    shutil.copy(face_file, dest)
    return f"✅ Face reference uploaded: {os.path.basename(dest)}", get_face_ref_choices()

@torch.inference_mode()
def generate(input):
    v = input["input"]
    seed = v["seed"] or random.randint(0, 2**63)

    cur_unet, cur_clip = unet, clip

    # Apply LoRAs
    for name_key, str_key in [("lora1_name","lora1_strength"),("lora2_name","lora2_strength")]:
        n = v.get(name_key, "None")
        s = float(v.get(str_key, 1.0))
        if n and n != "None":
            cur_unet, cur_clip = LoraLoader.load_lora(cur_unet, cur_clip, n, s, s)

    pos = CLIPTextEncode.encode(cur_clip, v["positive_prompt"])[0]
    neg = CLIPTextEncode.encode(cur_clip, v["negative_prompt"])[0]
    lat = EmptyLatentImage.generate(v["width"], v["height"], batch_size=1)[0]
    samp = KSampler.sample(
        cur_unet, seed, v["steps"], v["cfg"],
        v["sampler_name"], v["scheduler"],
        pos, neg, lat, denoise=v["denoise"]
    )[0]
    decoded = VAEDecode.decode(vae, samp)[0].detach()
    gen_img = Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])

    # ── Face swap with ReActor ──
    face_ref = v.get("face_ref", "None")
    face_strength = float(v.get("face_strength", 0.85))

    if REACTOR_AVAILABLE and face_ref and face_ref != "None":
        try:
            ref_path = os.path.join(FACE_REF_DIR, face_ref)
            ref_img   = np.array(Image.open(ref_path).convert("RGB"))
            gen_array = np.array(gen_img)

            # ReActor: swap face from reference onto generated image
            result = ReActorFaceSwap.execute(
                input_image=gen_array,
                source_image=ref_img,
                swap_model="inswapper_128.onnx",
                facedetection="retinaface_resnet50",
                face_restore_model="GFPGANv1.4.pth",
                face_restore_visibility=face_strength,
                codeformer_weight=0.5,
                detect_gender_input="no",
                detect_gender_source="no",
                input_faces_index="0",
                source_faces_index="0",
                console_log_level=0,
            )
            if result and len(result) > 0:
                gen_img = Image.fromarray(result[0])
                print("✅ Face swap applied")
        except Exception as e:
            print(f"⚠️ Face swap failed: {e} — using generated image")

    save_path = get_save_path(v["positive_prompt"])
    gen_img.save(save_path)
    return save_path, seed


import gradio as gr

DEFAULT_POS = "siren, photorealistic, ultrareal, beautiful woman, detailed face, cinematic lighting"
DEFAULT_NEG = "low quality, blurry, bad anatomy, ugly, deformed, cartoon, anime"
ASPECTS = [
    "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1248x832 (3:2)", "832x1248 (2:3)", "1280x720 (16:9)", "720x1280 (9:16)"
]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 style=\\'text-align:center\\'>Z-Image Turbo + Face Lock</h1>")

    with gr.Row():
        with gr.Column():
            positive  = gr.Textbox(DEFAULT_POS, label="Positive Prompt", lines=4)
            negative  = gr.Textbox(DEFAULT_NEG, label="Negative Prompt", lines=2)
            aspect    = gr.Dropdown(ASPECTS, value="1024x1024 (1:1)", label="Aspect Ratio")

            with gr.Row():
                seed  = gr.Number(value=0, label="Seed (0=random)", precision=0)
                steps = gr.Slider(4, 25, value=9, step=1, label="Steps")
                cfg   = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")

            run = gr.Button("🚀 Generate", variant="primary")

            # ── Style LoRA ──
            with gr.Accordion("🎨 Style LoRA (NiceGirls)", open=True):
                with gr.Row():
                    lora1_select   = gr.Dropdown(get_lora_choices(), value="None", label="Style LoRA")
                    lora1_refresh  = gr.Button("🔄", scale=0)
                lora1_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Strength")
                lora1_upload   = gr.File(label="Upload LoRA", file_types=[".safetensors"])
                lora1_btn      = gr.Button("📤 Upload")
                lora1_status   = gr.Textbox(interactive=False, label="Status")
                lora1_btn.click(upload_lora, [lora1_upload], [lora1_status, lora1_select])
                lora1_refresh.click(lambda: gr.Dropdown(choices=get_lora_choices()), outputs=[lora1_select])

            # ── Face Lock ──
            with gr.Accordion("🔒 Face Lock (ReActor)", open=True):
                gr.Markdown("Upload your character face image — ReActor will swap it onto every generation.")
                with gr.Row():
                    face_select  = gr.Dropdown(get_face_ref_choices(), value="None", label="Face Reference")
                    face_refresh = gr.Button("🔄", scale=0)
                face_strength  = gr.Slider(0.5, 1.0, value=0.85, step=0.05,
                                           label="Face Lock Strength (0.85 = natural, 1.0 = exact)")
                face_upload    = gr.File(label="Upload Face Reference", file_types=[".jpg",".png",".jpeg"])
                face_btn       = gr.Button("📤 Upload Face")
                face_status    = gr.Textbox(interactive=False, label="Status")
                face_btn.click(upload_face_ref, [face_upload], [face_status, face_select])
                face_refresh.click(lambda: gr.Dropdown(choices=get_face_ref_choices()), outputs=[face_select])

            # ── Optional 2nd LoRA ──
            with gr.Accordion("➕ Extra LoRA (optional)", open=False):
                with gr.Row():
                    lora2_select  = gr.Dropdown(get_lora_choices(), value="None", label="LoRA 2")
                    lora2_refresh = gr.Button("🔄", scale=0)
                lora2_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Strength")
                lora2_refresh.click(lambda: gr.Dropdown(choices=get_lora_choices()), outputs=[lora2_select])

        with gr.Column():
            output_img    = gr.Image(label="Generated Image", height=500)
            download_img  = gr.File(label="Download")
            used_seed     = gr.Textbox(label="Seed Used", show_copy_button=True)

    def run_fn(pos, neg, aspect, seed, steps, cfg,
               lora1, lora1_str, face_ref, face_str,
               lora2, lora2_str):
        w, h = [int(x) for x in aspect.split("(")[0].strip().split("x")]
        return generate({"input": {
            "positive_prompt": pos, "negative_prompt": neg,
            "width": w, "height": h, "batch_size": 1,
            "seed": int(seed), "steps": int(steps),
            "cfg": float(cfg), "sampler_name": "euler",
            "scheduler": "simple", "denoise": 1.0,
            "lora1_name": lora1, "lora1_strength": lora1_str,
            "lora2_name": lora2, "lora2_strength": lora2_str,
            "face_ref": face_ref, "face_strength": face_str,
        }})

    run.click(
        fn=run_fn,
        inputs=[positive, negative, aspect, seed, steps, cfg,
                lora1_select, lora1_strength,
                face_select, face_strength,
                lora2_select, lora2_strength],
        outputs=[download_img, output_img, used_seed]
    )

demo.launch(share=True, debug=True)
'''

with open("/kaggle/working/ComfyUI/app.py", "w") as f:
    f.write(app_code)

print("✅ New app.py written with face lock support")
