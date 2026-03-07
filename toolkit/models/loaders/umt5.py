from typing import List
import torch
import os
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config
from safetensors.torch import load_file, save_file
from toolkit.models.loaders.comfy import get_comfy_path
from toolkit.paths import COMFY_MODELS_PATH
import tempfile
import shutil

# Known repos & identifiers
DEEPBEEP_REPO_ID = "DeepBeepMeep/Wan2.1"
NSFW_ZOOTKITTY_REPO_ID = "zootkitty/nsfw_wan_umt5-xxl_bf16_fixed"
UMT5_SUBFOLDER = "umt5-xxl"
DEEPBEEP_DOWNLOAD_URL = f"https://huggingface.co/{DEEPBEEP_REPO_ID}/resolve/main/{UMT5_SUBFOLDER}/models_t5_umt5-xxl-enc-bf16.safetensors"

def download_file(url: str, local_path: str, desc: str = "Downloading"):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(local_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def _normalize_umt5_path(model_path: str):
    if model_path in ("", None):
        return DEEPBEEP_REPO_ID, UMT5_SUBFOLDER
    if model_path == DEEPBEEP_REPO_ID:
        return DEEPBEEP_REPO_ID, UMT5_SUBFOLDER
    if model_path == NSFW_ZOOTKITTY_REPO_ID:
        return NSFW_ZOOTKITTY_REPO_ID, None
    if os.path.exists(model_path):
        if os.path.exists(os.path.join(model_path, 'umt5-xxl')):
            return os.path.join(model_path, 'umt5-xxl'), None
        return model_path, None
    if UMT5_SUBFOLDER in model_path or model_path.endswith("/" + UMT5_SUBFOLDER):
        return DEEPBEEP_REPO_ID, UMT5_SUBFOLDER
    return model_path, None

# ==================== KEY REMAPPING (only for DeepBeepMeep format) ====================
def remap_deepbeep_umt5_state_dict(state_dict: dict) -> dict:
    # (your current working remapping function - unchanged)
    print("Remapping DeepBeepMeep-style UMT5 keys...")
    new_sd = {}
    for key, value in state_dict.items():
        handled = False
        if key == "token_embedding.weight":
            new_sd["shared.weight"] = value
            new_sd["encoder.embed_tokens.weight"] = value
            handled = True
        if key == "norm.weight" or key == "final_layer_norm.weight":
            new_sd["encoder.final_layer_norm.weight"] = value
            handled = True
        if key.startswith("blocks."):
            parts = key.split(".")
            block_idx = parts[1]
            rest = ".".join(parts[2:])
            if rest.startswith("attn."):
                attn_part = rest[5:]
                new_key = f"encoder.block.{block_idx}.layer.0.SelfAttention.{attn_part}"
                new_sd[new_key] = value
                handled = True
            elif rest.startswith("ffn."):
                ffn_part = rest[4:]
                if ffn_part == "fc1.weight":
                    new_sd[f"encoder.block.{block_idx}.layer.1.DenseReluDense.wi_1.weight"] = value
                    handled = True
                elif ffn_part == "gate.0.weight":
                    new_sd[f"encoder.block.{block_idx}.layer.1.DenseReluDense.wi_0.weight"] = value
                    handled = True
                elif ffn_part == "fc2.weight":
                    new_sd[f"encoder.block.{block_idx}.layer.1.DenseReluDense.wo.weight"] = value
                    handled = True
            elif rest == "norm1.weight":
                new_sd[f"encoder.block.{block_idx}.layer.0.layer_norm.weight"] = value
                handled = True
            elif rest == "norm2.weight":
                new_sd[f"encoder.block.{block_idx}.layer.1.layer_norm.weight"] = value
                handled = True
            elif rest == "pos_embedding.embedding.weight":
                new_key = f"encoder.block.{block_idx}.layer.0.SelfAttention.relative_attention_bias.weight"
                new_sd[new_key] = value
                handled = True
        if not handled:
            print(f"Skipped key: {key}")
    return new_sd

# ==================== MAIN LOADER ====================
def get_umt5_encoder(
    model_path: str,
    tokenizer_subfolder: str = None,
    encoder_subfolder: str = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    comfy_files: List[str] = None,
) -> tuple[AutoTokenizer, UMT5EncoderModel]:
    if comfy_files is None:
        comfy_files = ["text_encoders/umt5_xxl_fp16.safetensors"]

    effective_path, detected_subfolder = _normalize_umt5_path(model_path)
    is_local_path = os.path.exists(effective_path)

    print(f"[DEBUG] UMT5 loading - path='{effective_path}', local={is_local_path}, dtype={torch_dtype}")

    # ──────────────────────────────────────────────────────────────
    #  NEW PRIORITY: zootkitty/nsfw_wan_umt5-xxl_bf16_fixed
    # ──────────────────────────────────────────────────────────────
    if NSFW_ZOOTKITTY_REPO_ID in str(effective_path) or \
       (is_local_path and any(f.endswith("nsfw_wan_umt5-xxl_bf16_fixed.safetensors") for f in os.listdir(effective_path) if f.endswith('.safetensors'))):

        print("[INFO] Detected zootkitty/nsfw_wan_umt5-xxl_bf16_fixed format (single safetensors with embedded spiece.model)")

        if is_local_path:
            for f in os.listdir(effective_path):
                if f == "nsfw_wan_umt5-xxl_bf16_fixed.safetensors" or "nsfw_wan_umt5" in f.lower():
                    safetensors_path = os.path.join(effective_path, f)
                    break
            else:
                raise FileNotFoundError("nsfw_wan_umt5-xxl_bf16_fixed.safetensors not found in local path")
        else:
            cache_dir = os.path.expanduser("~/.cache/umt5_nsfw_zootkitty")
            os.makedirs(cache_dir, exist_ok=True)
            cached_file = os.path.join(cache_dir, "nsfw_wan_umt5-xxl_bf16_fixed.safetensors")
            if not os.path.exists(cached_file):
                url = f"https://huggingface.co/{NSFW_ZOOTKITTY_REPO_ID}/resolve/main/nsfw_wan_umt5-xxl_bf16_fixed.safetensors"
                print("[INFO] Downloading zootkitty NSFW UMT5 (cached)...")
                download_file(url, cached_file)
            safetensors_path = cached_file

        state_dict = load_file(safetensors_path, device="cpu")

        # Extract spiece.model if present
        temp_dir = tempfile.mkdtemp(prefix="umt5_nsfw_tokenizer_")
        spiece_path = os.path.join(temp_dir, "spiece.model")

        if "spiece_model" in state_dict:
            print("[INFO] Found embedded spiece.model → extracting...")
            spiece_bytes = state_dict["spiece_model"]
            if isinstance(spiece_bytes, torch.Tensor):
                spiece_bytes = spiece_bytes.cpu().numpy().tobytes()
            with open(spiece_path, "wb") as f:
                f.write(spiece_bytes)
        else:
            raise ValueError("No 'spiece_model' key found in state_dict — cannot create tokenizer")

        # Remove spiece from state dict so it doesn't confuse load_state_dict
        state_dict.pop("spiece_model", None)

        # Load tokenizer from extracted spiece
        tokenizer = AutoTokenizer.from_pretrained(temp_dir)

        # Load model (keys should already be in HF format)
        config = UMT5Config.from_pretrained("google/umt5-xxl")
        text_encoder = UMT5EncoderModel(config).to(dtype=torch_dtype)
        missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False)

        print(f"[INFO] zootkitty NSFW load → missing: {len(missing)}, unexpected: {len(unexpected)}")
        if missing or unexpected:
            print("Missing:", missing)
            print("Unexpected:", unexpected)

        # Cleanup temporary directory after loading
        shutil.rmtree(temp_dir, ignore_errors=True)

        return tokenizer, text_encoder

    # ──────────────────────────────────────────────────────────────
    #  1. Old ai-toolkit structure
    # ──────────────────────────────────────────────────────────────
    if is_local_path and os.path.exists(os.path.join(effective_path, "text_encoder", "config.json")):
        print("[INFO] Detected ai-toolkit/umt5_xxl_encoder structure")
        tokenizer = AutoTokenizer.from_pretrained(effective_path, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            effective_path, subfolder="text_encoder",
            torch_dtype=torch_dtype, low_cpu_mem_usage=True, ignore_mismatched_sizes=True
        )
        return tokenizer, text_encoder

    # ──────────────────────────────────────────────────────────────
    #  2. DeepBeepMeep native format (fallback)
    # ──────────────────────────────────────────────────────────────
    safetensors_path = None
    if is_local_path:
        for f in os.listdir(effective_path):
            if f.endswith('.safetensors') and ('umt5' in f.lower() or 't5' in f.lower()):
                safetensors_path = os.path.join(effective_path, f)
                break

    if safetensors_path or DEEPBEEP_REPO_ID in str(effective_path):
        print(f"[INFO] Loading DeepBeepMeep/Wan2.1 native format")
        if safetensors_path and os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path, device="cpu")
        else:
            cache_dir = os.path.expanduser("~/.cache/umt5_wan")
            os.makedirs(cache_dir, exist_ok=True)
            cached_file = os.path.join(cache_dir, "models_t5_umt5-xxl-enc-bf16.safetensors")
            if not os.path.exists(cached_file):
                print("[INFO] Downloading DeepBeepMeep UMT5...")
                download_file(DEEPBEEP_DOWNLOAD_URL, cached_file)
            state_dict = load_file(cached_file, device="cpu")

        state_dict = remap_deepbeep_umt5_state_dict(state_dict)
        config = UMT5Config.from_pretrained("google/umt5-xxl")
        text_encoder = UMT5EncoderModel(config).to(dtype=torch_dtype)
        missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False)
        print(f"[INFO] DeepBeep remapped load → missing: {len(missing)}, unexpected: {len(unexpected)}")

        tokenizer = AutoTokenizer.from_pretrained(
            effective_path if is_local_path else DEEPBEEP_REPO_ID,
            subfolder=UMT5_SUBFOLDER if not is_local_path else None
        )
        return tokenizer, text_encoder

    # ──────────────────────────────────────────────────────────────
    #  3. Generic fallback
    # ──────────────────────────────────────────────────────────────
    print("[INFO] Falling back to standard HF loading")
    tokenizer = AutoTokenizer.from_pretrained(effective_path, subfolder=tokenizer_subfolder)
    text_encoder = UMT5EncoderModel.from_pretrained(
        effective_path, subfolder=encoder_subfolder,
        torch_dtype=torch_dtype, ignore_mismatched_sizes=True
    )
    return tokenizer, text_encoder