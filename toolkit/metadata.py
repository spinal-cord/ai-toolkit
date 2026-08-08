import json
from collections import OrderedDict
from io import BytesIO

import safetensors
from safetensors import safe_open

from info import software_meta
from toolkit.train_tools import addnet_hash_legacy
from toolkit.train_tools import addnet_hash_safetensors


def get_meta_for_safetensors(meta: OrderedDict, name=None, add_software_info=True) -> OrderedDict:
    # stringify the meta and reparse OrderedDict to replace [name] with name
    meta_string = json.dumps(meta)
    if name is not None:
        meta_string = meta_string.replace("[name]", name)
    save_meta = json.loads(meta_string, object_pairs_hook=OrderedDict)
    if add_software_info:
        save_meta["software"] = software_meta
    # safetensors can only be one level deep
    for key, value in save_meta.items():
        # if not float, int, bool, or str, convert to json string
        if not isinstance(value, str):
            save_meta[key] = json.dumps(value)
    # add the pt format
    save_meta["format"] = "pt"
    return save_meta


def add_model_hash_to_meta(state_dict, meta: OrderedDict) -> OrderedDict:
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later.

    Includes protection against hangs on very large / complex state dicts
    (e.g., Wan 2.2 14B with high-rank LoRA and many long tensor names).
    """
    from toolkit.print import print_acc
    import time
    import threading

    print_acc("[METADATA] add_model_hash_to_meta() called")
    start_time = time.time()

    # Log state dict size and shape
    num_tensors = len(state_dict)
    total_bytes = sum(v.element_size() * v.numel() for v in state_dict.values())
    print_acc(f"[METADATA] State dict size: {total_bytes / 1024**3:.2f} GB, {num_tensors} tensors")

    # Analyze key names (safetensors header can become pathological with many long keys)
    key_lengths = [len(k) for k in state_dict.keys()]
    max_key_len = max(key_lengths) if key_lengths else 0
    total_key_len = sum(key_lengths)
    print_acc(
        f"[METADATA] Key stats: max_key_len={max_key_len}, "
        f"total_key_len={total_key_len:,}, avg_key_len={total_key_len / max(num_tensors, 1):.1f}"
    )

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in meta.items() if k.startswith("ss_")}

    # Decide whether hash computation looks "dangerous"
    # These thresholds are tuned to catch cases like Wan 2.2 14B with huge LoRA ranks.
    total_bytes_gb = total_bytes / (1024**3)
    dangerous = False
    reasons = []

    if total_bytes_gb > 5:
        dangerous = True
        reasons.append(f"state_dict_size={total_bytes_gb:.2f}GB")
    if num_tensors > 1500:
        dangerous = True
        reasons.append(f"tensor_count={num_tensors}")
    if total_key_len > 200_000:  # ~200k chars in keys total
        dangerous = True
        reasons.append(f"total_key_len={total_key_len:,}")

    if dangerous:
        print_acc(
            f"[METADATA] WARNING: Hash computation looks expensive "
            f"(reasons: {', '.join(reasons)}). "
            f"This can cause long pauses or apparent hangs in safetensors.torch.save()."
        )

    # Hard timeout for the entire safetensors.torch.save() call.
    # If it doesn't finish in time, we abort hash computation instead of hanging.
    save_timeout = 600  # 10 minutes
    save_done = threading.Event()
    save_error = [None]
    save_result = [None]

    def _do_save():
        try:
            print_acc("[METADATA] Starting safetensors.torch.save() for hash computation...")
            save_start = time.time()

            # Periodic logging helper (via a secondary daemon thread)
            save_start_ts = time.time()
            def _log_progress():
                while not save_done.is_set():
                    elapsed = time.time() - save_start_ts
                    if elapsed >= 120:  # start logging after 2 minutes
                        print_acc(f"[METADATA] safetensors.torch.save() still running... ({elapsed:.0f}s elapsed)")
                    time.sleep(30)

            progress_thread = threading.Thread(target=_log_progress, daemon=True)
            progress_thread.start()

            # For very large state dicts, this can be memory-intensive and CPU-bound:
            # it builds the entire file in memory as bytes, including metadata header.
            bytes_data = safetensors.torch.save(state_dict, metadata)

            save_time = time.time() - save_start
            save_result[0] = bytes_data
            save_error[0] = None
            print_acc(
                f"[METADATA] safetensors.torch.save() completed in {save_time:.2f}s. "
                f"Bytes size: {len(bytes_data) / 1024**3:.2f} GB"
            )
        except Exception as e:
            save_error[0] = e
            save_result[0] = None
            print_acc(f"[METADATA] safetensors.torch.save() failed: {e}")
        finally:
            save_done.set()

    save_thread = threading.Thread(target=_do_save, daemon=True)
    save_thread.start()

    # Wait with timeout
    finished = save_done.wait(timeout=save_timeout)

    if not finished:
        print_acc(
            f"[METADATA] ERROR: safetensors.torch.save() did not finish within {save_timeout}s. "
            f"Aborting hash computation to avoid hang. "
            f"Consider setting AITK_SKIP_MODEL_HASH=true for this training."
        )
        meta["sshs_model_hash"] = None
        meta["sshs_legacy_hash"] = None
        meta["ss_hash_status"] = "skipped_timeout"
        total_time = time.time() - start_time
        print_acc(f"[METADATA] add_model_hash_to_meta() aborted after {total_time:.2f}s (timeout)")
        return meta

    if save_error[0] is not None:
        print_acc(
            f"[METADATA] ERROR: safetensors.torch.save() raised an exception: {save_error[0]}. "
            f"Skipping hash computation."
        )
        meta["sshs_model_hash"] = None
        meta["sshs_legacy_hash"] = None
        meta["ss_hash_status"] = "skipped_error"
        total_time = time.time() - start_time
        print_acc(f"[METADATA] add_model_hash_to_meta() aborted after {total_time:.2f}s (error)")
        return meta

    bytes_data = save_result[0]

    # Clear the state_dict reference to help GC before hashing
    del state_dict

    b = BytesIO(bytes_data)

    print_acc("[METADATA] Computing model hashes...")
    hash_start = time.time()
    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    hash_time = time.time() - hash_start
    print_acc(f"[METADATA] Hash computation completed in {hash_time:.2f}s")

    # Clear bytes_data to help GC
    del bytes_data
    del b

    meta["sshs_model_hash"] = model_hash
    meta["sshs_legacy_hash"] = legacy_hash
    meta["ss_hash_status"] = "ok"

    total_time = time.time() - start_time
    print_acc(f"[METADATA] add_model_hash_to_meta() completed in {total_time:.2f}s")
    return meta


def add_base_model_info_to_meta(
        meta: OrderedDict,
        base_model: str = None,
        is_v1: bool = False,
        is_v2: bool = False,
        is_xl: bool = False,
) -> OrderedDict:
    if base_model is not None:
        meta['ss_base_model'] = base_model
    elif is_v2:
        meta['ss_v2'] = True
        meta['ss_base_model_version'] = 'sd_2.1'

    elif is_xl:
        meta['ss_base_model_version'] = 'sdxl_1.0'
    else:
        # default to v1.5
        meta['ss_base_model_version'] = 'sd_1.5'
    return meta


def parse_metadata_from_safetensors(meta: OrderedDict) -> OrderedDict:
    parsed_meta = OrderedDict()
    for key, value in meta.items():
        try:
            parsed_meta[key] = json.loads(value)
        except json.decoder.JSONDecodeError:
            parsed_meta[key] = value
    return parsed_meta


def load_metadata_from_safetensors(file_path: str) -> OrderedDict:
    try:
        with safe_open(file_path, framework="pt") as f:
            metadata = f.metadata()
        return parse_metadata_from_safetensors(metadata)
    except Exception as e:
        print(f"Error loading metadata from {file_path}: {e}")
        return OrderedDict()
