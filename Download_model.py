#!/usr/bin/env python3
"""
download_qwen.py: Download Qwen2.5-7B-Instruct model for offline use
"""

import os
from huggingface_hub import snapshot_download

MODEL_DIR = "/gpfs/wolf2/olcf/trn040/scratch/8mn/project1/qwen2.5-7b"

# Ensure the target directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Optional: Set Hugging Face cache dir to scratch
os.environ["HF_HOME"] = "/gpfs/wolf2/olcf/trn040/scratch/8mn/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]

print("Downloading Qwen2.5-7B-Instruct...")

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,  # avoids quota issues with symlinks
    resume_download=True  # resume if partially downloaded
)

print(f"Model saved to {MODEL_DIR}")
