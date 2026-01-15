from typing import List
import torch
import os
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel
from toolkit.models.loaders.comfy import get_comfy_path
from toolkit.paths import COMFY_MODELS_PATH


# Download URL for the UMT5 text encoder model
# This is the zootkitty/wan_umt5-xxl_bf16_fixed repository's fixed bf16 version
# https://huggingface.co/zootkitty/wan_umt5-xxl_bf16_fixed/resolve/main/nsfw_wan_umt5-xxl_bf16_fixed.safetensors?download=true
UMT5_DOWNLOAD_URL = "https://huggingface.co/zootkitty/nsfw_wan_umt5-xxl_bf16_fixed/resolve/main/nsfw_wan_umt5-xxl_bf16_fixed.safetensors?download=true"


def download_file(url: str, local_path: str, desc: str = "Downloading"):
    """
    Download a file from a URL to a local path with a progress bar.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def get_umt5_encoder(
    model_path: str,
    tokenizer_subfolder: str = None,
    encoder_subfolder: str = None,
    torch_dtype: str = torch.bfloat16,
    comfy_files: List[str] = [
        "text_encoders/umt5_xxl_fp16.safetensors",
        "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ],
) -> UMT5EncoderModel:
    """
    Load the UMT5 encoder model from the specified path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder=tokenizer_subfolder)
    comfy_path = get_comfy_path(comfy_files)
    
    # If the file doesn't exist locally, download it from the URL
    if comfy_path is None and COMFY_MODELS_PATH is not None:
        # Check for the first file in the list (umt5_xxl_fp16.safetensors)
        target_filename = comfy_files[0] if comfy_files else "text_encoders/umt5_xxl_fp16.safetensors"
        local_file_path = os.path.join(COMFY_MODELS_PATH, target_filename)
        
        if not os.path.exists(local_file_path):
            print(f"Downloading UMT5 encoder from {UMT5_DOWNLOAD_URL}")
            print(f"Saving to {local_file_path}")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            download_file(UMT5_DOWNLOAD_URL, local_file_path, desc="Downloading UMT5 encoder")
            print("Download complete!")
        
        if os.path.exists(local_file_path):
            comfy_path = local_file_path
    
    if comfy_path is not None:
        text_encoder = UMT5EncoderModel.from_single_file(
            comfy_path, torch_dtype=torch_dtype
        )
    else:
        print(f"Using {model_path} for UMT5 encoder.")
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_path, subfolder=encoder_subfolder, torch_dtype=torch_dtype
        )
    return tokenizer, text_encoder

