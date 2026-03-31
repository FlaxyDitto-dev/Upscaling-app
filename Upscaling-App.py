import os
import warnings
import logging
import torch
import requests
from io import BytesIO
from PIL import Image
from aura_sr import AuraSR

# 1. Total Silence
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Optimization
torch.set_float32_matmul_precision('high')

def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

print("Loading AuraSR model...")
aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

# RTX 5060 Ti Speed Setup
device = "cuda"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Input Processing
url = input("Enter the input image's URL: ")
raw_image = load_image_from_url(url)

# Force 512x512 input
image = raw_image.resize((512, 512), Image.LANCZOS)
image.save("in-test.png")
print(f"Input: {image.size} | Final Target: 4096x4096")

# 2. Optimized Multi-Pass Inference
with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype):
    
    # PASS 1: 512 -> 2048 (4x)
    print("Pass 1: AI Upscale (4x)...")
    upscaled_4x = aura_sr.upscale_4x_overlapped(image)

    # STEP 2: Downscale by 2 (2048 -> 1024)
    print("Intermediate: Downscaling 2x...")
    downscaled_2x = upscaled_4x.resize((1024, 1024), Image.LANCZOS)

    # PASS 3: 1024 -> 4096 (Final 4x)
    print("Pass 2: AI Upscale (Final 4x)...")
    final_output = aura_sr.upscale_4x_overlapped(downscaled_2x)

# Save
final_output.save("out-test.png")
print(f"Success! Final size: {final_output.size}. Saved as 'out-test.png'")
