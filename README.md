# AuraSR Ultra-Fast 8x Upscaler

A high-performance Python script for upscaling images to 4096x4096px using the AuraSR-v2 GigaGAN-based model. Optimized for RTX 50-series GPUs.

## 🚀 System Requirements
- GPU: NVIDIA RTX 30/40/50 Series (Optimized for RTX 5060 Ti 16GB)
- VRAM: 8GB+ (Uses ~5GB peak)
- Python: 3.10+
- Drivers: CUDA 12.1+ recommended

## 📦 Installation
Run the following command to install the required libraries:
pip install torch torchvision torchaudio --index-url https://pytorch.org
pip install aura-sr pillow requests

## 🛠️ Configuration

### 1. Fixing the "Download Glitch"
The first time you run the script, it needs to download the model weights (~2.5GB).
- First Run: Set os.environ["HF_HUB_OFFLINE"] = "0" at the top of the script.
- Subsequent Runs: Set os.environ["HF_HUB_OFFLINE"] = "1" to skip the check and start instantly.

### 2. Changing Resolutions
To customize the output, look for these lines in the script:
- Initial Resize: image = raw_image.resize((512, 512)) 
  Change (512, 512) to adjust the starting quality.
- Pass 2 Downscale (/2): downscaled_2x = upscaled_4x.resize((1024, 1024))
  Change (1024, 1024) to adjust the intermediate detail density.
- Final Output: The final AI pass automatically turns the 1024px image into a 4096px image (4x).

## ⚡ Key Optimizations
- BFloat16: Native speed boost for RTX 5060 Ti.
- SDPA Flash Attention: Faster processing with less VRAM.
- Silent Mode: Suppresses all "FutureWarnings" and HuggingFace logs.
