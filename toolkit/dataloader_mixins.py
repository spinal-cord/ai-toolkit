import base64
import glob
import hashlib
import json
import math
import os
import random
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Dict, Union, Any
import traceback

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipImageProcessor

from toolkit.audio.preserve_pitch import time_stretch_preserve_pitch
from toolkit.basic import flush, value_map, get_resize_method
from toolkit.buckets import get_bucket_for_image_size, get_resolution
from toolkit.config_modules import ControlTypes
from toolkit.control_generator import ControlGenerator
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.pixtral_vision import PixtralVisionImagePreprocessorCompatible
from toolkit.prompt_utils import inject_trigger_into_prompt
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
import albumentations as A
from toolkit.print import print_acc
from toolkit.accelerator import get_accelerator
from toolkit.prompt_utils import PromptEmbeds
from torchvision.transforms import functional as TF

from toolkit.train_tools import get_torch_dtype
from toolkit import dataset_crypto

if TYPE_CHECKING:
    from toolkit.data_loader import AiToolkitDataset
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.stable_diffusion_model import StableDiffusion

accelerator = get_accelerator()

# def get_associated_caption_from_img_path(img_path):
# https://demo.albumentations.ai/
class Augments:
    def __init__(self, **kwargs):
        self.method_name = kwargs.get('method', None)
        self.params = kwargs.get('params', {})

        # convert kwargs enums for cv2
        for key, value in self.params.items():
            if isinstance(value, str):
                # split the string
                split_string = value.split('.')
                if len(split_string) == 2 and split_string[0] == 'cv2':
                    if hasattr(cv2, split_string[1]):
                        self.params[key] = getattr(cv2, split_string[1].upper())
                    else:
                        raise ValueError(f"invalid cv2 enum: {split_string[1]}")


transforms_dict = {
    'ColorJitter': transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
    'RandomEqualize': transforms.RandomEqualize(p=0.2),
}

img_ext_list = ['.jpg', '.jpeg', '.png', '.webp', '.jxl']


def standardize_images(images):
    """
    Standardize the given batch of images using the specified mean and std.
    Expects values of 0 - 1

    Args:
    images (torch.Tensor): A batch of images in the shape of (N, C, H, W),
                           where N is the number of images, C is the number of channels,
                           H is the height, and W is the width.

    Returns:
    torch.Tensor: Standardized images.
    """
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    # Define the normalization transform
    normalize = transforms.Normalize(mean=mean, std=std)

    # Apply normalization to each image in the batch
    standardized_images = torch.stack([normalize(img) for img in images])

    return standardized_images

def clean_caption(caption):
    # this doesnt make any sense anymore in a world that is not based on comma seperated tokens
    # # remove any newlines
    # caption = caption.replace('\n', ', ')
    # # remove new lines for all operating systems
    # caption = caption.replace('\r', ', ')
    # caption_split = caption.split(',')
    # # remove empty strings
    # caption_split = [p.strip() for p in caption_split if p.strip()]
    # # join back together
    # caption = ', '.join(caption_split)
    return caption


def _filter_prompts_by_mode(prompts: List[Dict[str, Any]], is_i2v_mode: bool = True, log_warning: bool = True) -> List[Dict[str, Any]]:
    """
    Filter prompts based on the current training mode (is_i2v_mode).

    - If is_i2v_mode is True (I2V training), include prompts where do_i2v is True
    - If is_i2v_mode is False (T2V training), include prompts where do_t2v is True
    - A prompt is included if it matches the current training mode

    This ensures that when dataset is doubled (both I2V and T2V enabled),
    each copy only receives prompts appropriate for its mode.

    Args:
        prompts: List of prompt objects with do_i2v/do_t2v flags
        is_i2v_mode: True for I2V training, False for T2V training
        log_warning: If True, log a generic warning when all prompts are filtered out
    """
    import logging
    logger = logging.getLogger(__name__)

    filtered = []
    for prompt in prompts:
        prompt_do_i2v = prompt.get('do_i2v', True)
        prompt_do_t2v = prompt.get('do_t2v', True)

        if is_i2v_mode:
            if prompt_do_i2v:
                filtered.append(prompt)
        else:
            if prompt_do_t2v:
                filtered.append(prompt)

    # Warn if all prompts were filtered out (generic warning, no file/content details)
    if log_warning and len(prompts) > 0 and len(filtered) == 0:
        mode_name = 'I2V' if is_i2v_mode else 'T2V'
        logger.warning(
            f"All prompts filtered out for {mode_name} mode on a JSON-captions dataset item. "
            f"This will result in an empty caption. Check do_i2v/do_t2v flags in your JSON captions."
        )

    return filtered


def get_mode_from_config(dataset_config) -> str:
    """
    Determine the training mode from dataset config.

    Returns:
        'i2v' if do_i2v=True and do_t2v=False
        't2v' if do_i2v=False and do_t2v=True
        'both' if both are True (should only happen in doubled DTO path)
        'i2v' as default if neither is explicitly set
    """
    do_i2v = getattr(dataset_config, 'do_i2v', True)
    do_t2v = getattr(dataset_config, 'do_t2v', False)

    if do_i2v and not do_t2v:
        return 'i2v'
    elif not do_i2v and do_t2v:
        return 't2v'
    elif do_i2v and do_t2v:
        return 'both'
    else:
        # Default to I2V if config is ambiguous
        return 'i2v'


def is_i2v_mode_from_config(dataset_config) -> bool:
    """
    Check if we're in I2V mode based on dataset config.
    Returns True if do_i2v=True and do_t2v=False (or default).
    """
    return get_mode_from_config(dataset_config) == 'i2v'


def detect_json_caption(path_no_ext: str) -> str:
    """
    Check if a JSON caption file exists alongside the image/video file.
    Returns the path to the JSON file if it exists, else None.
    """
    json_path = path_no_ext + '.json'
    if os.path.exists(json_path):
        return json_path
    return None


def compute_json_file_hash(json_path: str) -> str:
    """
    Compute MD5 hash of JSON file contents for cache invalidation.
    Returns a base64-encoded, URL-safe hash string.
    """
    with open(json_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash


def parse_json_captions(json_path: str) -> List[Dict[str, Any]]:
    """
    Parse a JSON caption file and return a list of prompt objects.
    
    Expected JSON structure:
    [
        {
            "prompt": "string",
            "weight": float (optional),
            "do_i2v": bool (optional),
            "do_t2v": bool (optional)
        },
        ...
    ]
    
    Or a single object (will be wrapped in a list):
    {
        "prompt": "string",
        "weight": float (optional),
        ...
    }
    
    Field defaults (applied when missing, None, wrong type, or invalid value):
    - prompt: ''
    - do_i2v: True
    - do_t2v: True
    - weight: None (auto-computed if missing/invalid)
    
    Weight normalization:
    - If weight is missing, None, or negative, it will be computed automatically
    - If all prompts lack weights, they get equal chance
    - If only one prompt exists, weight is set to 1.0
    """
    raw_content = dataset_crypto.read_text_file(json_path)
    data = json.loads(raw_content)
    
    # Handle single object (not in a list)
    if isinstance(data, dict):
        data = [data]
    
    # Ensure it's a list
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON to be a list or object, got {type(data)}")
    
    prompts = []
    for item in data:
        if not isinstance(item, dict):
            continue
        
        # Safely extract prompt, default to '' if missing, None, or not a string
        raw_prompt = item.get('prompt')
        prompt_text = raw_prompt if isinstance(raw_prompt, str) else ''
        
        # Safely extract do_i2v, default to True if missing, None, wrong type, or invalid
        raw_i2v = item.get('do_i2v')
        is_i2v = raw_i2v is True if isinstance(raw_i2v, bool) else True
        
        # Safely extract do_t2v, default to True if missing, None, wrong type, or invalid
        raw_t2v = item.get('do_t2v')
        is_t2v = raw_t2v is True if isinstance(raw_t2v, bool) else True
        
        # Safely extract weight
        raw_weight = item.get('weight')
        # Respect explicit 0 weight, treat missing/invalid as None (auto)
        weight = raw_weight if isinstance(raw_weight, (int, float)) and raw_weight >= 0 else None
        
        prompts.append({
            'prompt': prompt_text,
            'weight': weight,
            'do_i2v': is_i2v,
            'do_t2v': is_t2v,
        })
    
    # Empty prompts are kept as valid prompts.
    # They are cached separately and selected according to their weight.
    # This allows intentional use of empty captions (equivalent to caption dropout)
    # by specifying an empty prompt with a non-zero weight.
    
    if not prompts:
        return []
    
    # Normalize weights
    has_positive_weight = any(p['weight'] is not None and p['weight'] > 0 for p in prompts)
    
    if len(prompts) == 1:
        # Single prompt: default to 1.0 unless explicitly set to 0
        if prompts[0]['weight'] is None or prompts[0]['weight'] > 0:
            prompts[0]['weight'] = 1.0
        # If explicitly 0, keep it as is
    elif not has_positive_weight:
        # No positive weights found
        for p in prompts:
            if p['weight'] is None:
                p['weight'] = 1.0  # Auto-compute for missing weights
            # Explicit 0 weights are preserved
    else:
        # Some positive weights exist
        for p in prompts:
            if p['weight'] is None:
                p['weight'] = 1.0  # Default for missing weights
            # Explicit 0 weights are preserved
    
    return prompts


def select_prompt_weighted(prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select a prompt from the list using weighted random selection.
    Returns the selected prompt object.

    IMPORTANT: This selection is intentionally NON-DETERMINISTIC. The same image/video
    may get different prompts across training steps/epochs. This is by design to provide
    caption augmentation and help the model learn from multiple descriptions of the same
    media. If you need deterministic training, use .txt caption files instead of JSON
    with multiple prompts.

    Behavior when all weights are 0:
        Returns an empty caption (equivalent to caption dropout). This allows you to
        intentionally "disable" prompts by setting their weight to 0.
    """
    if not prompts:
        return {'prompt': '', 'weight': 0.0, 'do_i2v': True, 'do_t2v': True}
    
    if len(prompts) == 1:
        return prompts[0]
    
    # Extract weights for selection
    weights = [p['weight'] for p in prompts]
    total_weight = sum(weights)
    
    # If all weights are 0 (or negative), return empty caption.
    # This is intentional: setting all weights to 0 means "use empty caption".
    if total_weight <= 0:
        return {'prompt': '', 'weight': 0.0, 'do_i2v': True, 'do_t2v': True}
    
    # Weighted random selection
    rand = random.random() * total_weight
    cumulative = 0
    for prompt in prompts:
        cumulative += prompt['weight']
        if rand <= cumulative:
            return prompt
    
    # Fallback to last prompt
    return prompts[-1]

def waveform_to_stereo(waveform):
    c = waveform.shape[0]
    if c == 2:
        return waveform
    if c == 1:
        return waveform.expand(2, -1)
    if c == 6:  # 5.1: FL, FR, FC, LFE, BL, BR
        fl, fr, fc, _, bl, br = waveform
        k = 0.7071
        return torch.stack([fl + k * fc + k * bl, fr + k * fc + k * br])
    if c == 8:  # 7.1: FL, FR, FC, LFE, BL, BR, SL, SR
        fl, fr, fc, _, bl, br, sl, sr = waveform
        k = 0.7071
        return torch.stack([fl + k * fc + k * (bl + sl), fr + k * fc + k * (br + sr)])
    return waveform.mean(0, keepdim=True).expand(2, -1)


class CaptionMixin:
    def get_caption_item(self: 'AiToolkitDataset', index):
        if not hasattr(self, 'caption_type'):
            raise Exception('caption_type not found on class instance')
        if not hasattr(self, 'file_list'):
            raise Exception('file_list not found on class instance')
        img_path_or_tuple = self.file_list[index]
        ext = self.dataset_config.caption_ext
        if isinstance(img_path_or_tuple, tuple):
            img_path = img_path_or_tuple[0] if isinstance(img_path_or_tuple[0], str) else img_path_or_tuple[0].path
            # check if either has a prompt file
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = None
            prompt_path = path_no_ext + ext
        else:
            img_path = img_path_or_tuple if isinstance(img_path_or_tuple, str) else img_path_or_tuple.path
            # see if prompt file exists
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = path_no_ext + ext
                
        # allow folders to have a default prompt
        default_prompt_path = os.path.join(os.path.dirname(img_path), 'default.txt')
        default_prompt_path_with_ext = os.path.join(os.path.dirname(img_path), 'default' + ext)

        # Check for JSON caption file first (automatic detection, independent of caption_ext)
        json_caption_path = detect_json_caption(path_no_ext)
        
        if json_caption_path:
            # Parse JSON captions
            raw_prompts = parse_json_captions(json_caption_path)
            
            if raw_prompts:
                # Determine training mode from dataset config
                # Note: self is the dataset instance here, not FileItemDTO
                # For legacy paths, we use dataset_config.do_i2v/do_t2v to determine mode
                # For doubled datasets, this path is not used (DTO path handles it)
                mode = get_mode_from_config(self.dataset_config)
                # When both I2V and T2V are enabled in legacy path, treat as I2V
                # (legacy path doesn't support doubling, so we default to I2V mode)
                is_i2v = (mode in ('i2v', 'both'))
                
                filtered_prompts = _filter_prompts_by_mode(raw_prompts, is_i2v_mode=is_i2v)
                
                if filtered_prompts:
                    # Select a prompt using weighted random selection and apply clean_caption
                    selected = select_prompt_weighted(filtered_prompts)
                    prompt = clean_caption(selected['prompt'])
                else:
                    # Intentionally empty: JSON caption files provide their own
                    # caption logic (mode filtering, weighted selection, empty
                    # results when no prompts match). No default_prompt fallback
                    # is applied because the JSON file is the explicit source of
                    # truth and the user controls what happens when no prompts
                    # match the current mode.
                    prompt = ''
            else:
                prompt = ''
        elif os.path.exists(prompt_path):
            prompt = clean_caption(dataset_crypto.read_text_file(prompt_path))
        elif os.path.exists(default_prompt_path_with_ext):
            prompt = clean_caption(dataset_crypto.read_text_file(default_prompt_path_with_ext))
        elif os.path.exists(default_prompt_path):
            prompt = clean_caption(dataset_crypto.read_text_file(default_prompt_path))
        else:
            prompt = ''
            # get default_prompt if it exists on the class instance
            if hasattr(self, 'default_prompt'):
                prompt = self.default_prompt
            if hasattr(self, 'default_caption'):
                prompt = self.default_caption

        # handle replacements
        replacement_list = self.dataset_config.replacements if isinstance(self.dataset_config.replacements, list) else []
        for replacement in replacement_list:
            from_string, to_string = replacement.split('|')
            prompt = prompt.replace(from_string, to_string)

        return prompt


if TYPE_CHECKING:
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO


class Bucket:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.file_list_idx: List[int] = []


class BucketsMixin:
    def __init__(self):
        self.buckets: Dict[str, Bucket] = {}
        self.batch_indices: List[List[int]] = []

    def build_batch_indices(self: 'AiToolkitDataset'):
        self.batch_indices = []
        for key, bucket in self.buckets.items():
            for start_idx in range(0, len(bucket.file_list_idx), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(bucket.file_list_idx))
                batch = bucket.file_list_idx[start_idx:end_idx]
                # if the bucket has fewer items left than the requested batch size,
                # duplicate items from this batch to pad it up to batch_size
                if len(batch) < self.batch_size and len(batch) > 0:
                    pad = [batch[i % len(batch)] for i in range(self.batch_size - len(batch))]
                    batch = batch + pad
                self.batch_indices.append(batch)

    def shuffle_buckets(self: 'AiToolkitDataset'):
        for key, bucket in self.buckets.items():
            random.shuffle(bucket.file_list_idx)

    def setup_buckets(self: 'AiToolkitDataset', quiet=False):
        if not hasattr(self, 'file_list'):
            raise Exception(f'file_list not found on class instance {self.__class__.__name__}')
        if not hasattr(self, 'dataset_config'):
            raise Exception(f'dataset_config not found on class instance {self.__class__.__name__}')

        if self.epoch_num > 0:
            # no need to rebuild buckets for now
            # todo handle random cropping for buckets
            return
        self.buckets = {}  # clear it

        config: 'DatasetConfig' = self.dataset_config
        resolution = config.resolution
        bucket_tolerance = config.bucket_tolerance
        file_list: List['FileItemDTO'] = self.file_list

        # for file_item in enumerate(file_list):
        for idx, file_item in enumerate(file_list):
            file_item: 'FileItemDTO' = file_item
            if self.is_audio_model:
                bucket_key = f"{file_item.width}ms"
                if bucket_key not in self.buckets:
                    self.buckets[bucket_key] = Bucket(file_item.width, 1)
                self.buckets[bucket_key].file_list_idx.append(idx)
                continue
            width = int(file_item.width * file_item.dataset_config.scale)
            height = int(file_item.height * file_item.dataset_config.scale)

            if self.dataset_config.square_crop:
                # we scale first so smallest size matches resolution
                scale_factor_x = resolution / width
                scale_factor_y = resolution / height
                scale_factor = max(scale_factor_x, scale_factor_y)
                file_item.scale_to_width = math.ceil(width * scale_factor)
                file_item.scale_to_height = math.ceil(height * scale_factor)
                file_item.crop_width = resolution
                file_item.crop_height = resolution
                if width > height:
                    file_item.crop_x = int(file_item.scale_to_width / 2 - resolution / 2)
                    file_item.crop_y = 0
                else:
                    file_item.crop_x = 0
                    file_item.crop_y = int(file_item.scale_to_height / 2 - resolution / 2)
            else:
                bucket_resolution = get_bucket_for_image_size(
                    width, height,
                    resolution=resolution,
                    divisibility=bucket_tolerance
                )

                # Calculate scale factors for width and height
                width_scale_factor = bucket_resolution["width"] / width
                height_scale_factor = bucket_resolution["height"] / height

                # Use the maximum of the scale factors to ensure both dimensions are scaled above the bucket resolution
                max_scale_factor = max(width_scale_factor, height_scale_factor)

                # round up
                file_item.scale_to_width = int(math.ceil(width * max_scale_factor))
                file_item.scale_to_height = int(math.ceil(height * max_scale_factor))

                file_item.crop_height = bucket_resolution["height"]
                file_item.crop_width = bucket_resolution["width"]

                new_width = bucket_resolution["width"]
                new_height = bucket_resolution["height"]

                if self.dataset_config.random_crop:
                    # random crop
                    crop_x = random.randint(0, file_item.scale_to_width - new_width)
                    crop_y = random.randint(0, file_item.scale_to_height - new_height)
                    file_item.crop_x = crop_x
                    file_item.crop_y = crop_y
                else:
                    # do central crop
                    file_item.crop_x = int((file_item.scale_to_width - new_width) / 2)
                    file_item.crop_y = int((file_item.scale_to_height - new_height) / 2)

                if file_item.crop_y < 0 or file_item.crop_x < 0:
                    print_acc('debug')

            # check if bucket exists, if not, create it
            bucket_key = f'{file_item.crop_width}x{file_item.crop_height}'
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = Bucket(file_item.crop_width, file_item.crop_height)
            self.buckets[bucket_key].file_list_idx.append(idx)

        # print the buckets
        self.shuffle_buckets()
        self.build_batch_indices()
        if not quiet:
            print_acc(f'Bucket sizes for {self.dataset_path}:')
            for key, bucket in self.buckets.items():
                print_acc(f'{key}: {len(bucket.file_list_idx)} files')
            print_acc(f'{len(self.buckets)} buckets made')


class CaptionProcessingDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
            self.raw_caption: str = None
            self.raw_caption_short: str = None
            self.caption: str = None
            self.caption_short: str = None
            
            # New fields for JSON caption support
            self.raw_prompts: List[Dict[str, Any]] = []  # List of prompt objects from JSON
            self.selected_prompt: Dict[str, Any] = None  # The currently selected prompt object
            self.json_caption_path: str = None  # Path to JSON caption file if used
            self.json_file_hash: str = None  # MD5 hash of JSON file for cache invalidation

            dataset_config: DatasetConfig = kwargs.get('dataset_config', None)
            self.extra_values: List[float] = dataset_config.extra_values
            self.trigger_word = dataset_config.trigger_word

    # todo allow for loading from sd-scripts style dict
    def load_caption(self: 'FileItemDTO', caption_dict: Union[dict, None]=None):
        if self.raw_caption is not None:
            # we already loaded it
            pass
        elif caption_dict is not None and self.path in caption_dict and "caption" in caption_dict[self.path]:
            self.raw_caption = caption_dict[self.path]["caption"]
            if 'caption_short' in caption_dict[self.path]:
                self.raw_caption_short = caption_dict[self.path]["caption_short"]
                if self.dataset_config.use_short_captions:
                    self.raw_caption = caption_dict[self.path]["caption_short"]
        else:
            # see if prompt file exists
            path_no_ext = os.path.splitext(self.path)[0]
            
            # Check for JSON caption file first (automatic detection, independent of caption_ext)
            json_caption_path = detect_json_caption(path_no_ext)
            
            if json_caption_path:
                # Store JSON caption path and compute hash for cache invalidation
                self.json_caption_path = json_caption_path
                self.json_file_hash = compute_json_file_hash(json_caption_path)
                
                # Parse JSON captions
                self.raw_prompts = parse_json_captions(json_caption_path)
                
                if self.raw_prompts:
                    # Filter prompts based on training mode (is_i2v_mode)
                    is_i2v = getattr(self, 'is_i2v_mode', True)
                    filtered_prompts = _filter_prompts_by_mode(self.raw_prompts, is_i2v_mode=is_i2v)
                    
                    if filtered_prompts:
                        # Select a prompt using weighted random selection and apply clean_caption
                        # NOTE: This selection is intentionally NON-DETERMINISTIC for caption augmentation.
                        # With text embedding caching, a random prompt is selected at training time
                        # and its cached embedding is loaded.
                        self.selected_prompt = select_prompt_weighted(filtered_prompts)
                        self.raw_caption = clean_caption(self.selected_prompt['prompt'])
                    else:
                        # No prompts match the current mode, use empty caption
                        # Intentionally no default_prompt fallback: the JSON file
                        # is the explicit source of truth and the user controls
                        # what happens when no prompts match the current mode.
                        self.raw_caption = ''
                        self.selected_prompt = {'prompt': '', 'weight': 0.0, 'do_i2v': True, 'do_t2v': True}
                else:
                    self.raw_caption = ''
                    self.selected_prompt = {'prompt': '', 'weight': 0.0, 'do_i2v': True, 'do_t2v': True}
                
                # Set short caption to the same as main caption for JSON mode
                self.raw_caption_short = self.raw_caption
            else:
                # Fall back to existing .txt behavior
                prompt_ext = self.dataset_config.caption_ext
                prompt_path = path_no_ext + prompt_ext
                short_caption = None

                if os.path.exists(prompt_path):
                    # decrypted in RAM on the fly if the dataset is encrypted
                    prompt = dataset_crypto.read_text_file(prompt_path)
                    short_caption = None
                    prompt = clean_caption(prompt)
                    if short_caption is not None:
                        short_caption = clean_caption(short_caption)
                    # JSON mode does not apply default_caption fallback;
                    # this branch only runs for .txt fallback, which does.

                    if prompt.strip() == '' and self.dataset_config.default_caption is not None:
                        prompt = self.dataset_config.default_caption
                else:
                    prompt = ''
                    if self.dataset_config.default_caption is not None:
                        prompt = self.dataset_config.default_caption

                if short_caption is None:
                    short_caption = self.dataset_config.default_caption
                self.raw_caption = prompt
                self.raw_caption_short = short_caption
                # Initialize JSON-specific fields for non-JSON captions
                self.raw_prompts = []
                self.selected_prompt = None

        self.caption = self.get_caption()
        if self.raw_caption_short is not None:
            self.caption_short = self.get_caption(short_caption=True)
        
        # Clean up legacy per-file text embedding caches (no longer referenced
        # by the content-addressed cache) to save disk space.
        self.cleanup_legacy_text_embedding_caches()
    
    def _filter_prompts_by_mode(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter prompts based on the current training mode (is_i2v_mode).
        Delegates to the module-level _filter_prompts_by_mode helper.
        """
        return _filter_prompts_by_mode(prompts, is_i2v_mode=getattr(self, 'is_i2v_mode', True))
    
    def cleanup_legacy_text_embedding_caches(self: 'FileItemDTO'):
        """
        Clean up legacy per-file text embedding caches when a caption is loaded.

        Legacy caches were stored directly in ``_t_e_cache`` as
        ``{filename}_{hash}.safetensors`` (one file per dataset item per prompt).
        The current cache is content-addressed - files are named by the sha256
        of the prompt text and live in a versioned subdirectory of
        ``_t_e_cache`` - so legacy per-file files are never referenced anymore
        and can be deleted.

        Content-addressed cache files are intentionally NOT deleted here: they
        are shared across dataset items, and a prompt removed from one item's
        caption may still be referenced by another item.
        """
        img_dir = os.path.dirname(self.path)
        te_dir = os.path.join(img_dir, '_t_e_cache')

        if not os.path.exists(te_dir):
            return

        filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]

        # Find and delete legacy per-file cache files only
        deleted_count = 0
        for cache_file in os.listdir(te_dir):
            # Legacy format: {filename_no_ext}_{hash}.safetensors
            # (content-addressed files are named purely by the prompt's sha256)
            if cache_file.startswith(f'{filename_no_ext}_') and cache_file.endswith('.safetensors'):
                cache_path = os.path.join(te_dir, cache_file)
                if os.path.isfile(cache_path):
                    try:
                        os.remove(cache_path)
                        deleted_count += 1
                    except OSError:
                        pass  # Ignore errors deleting old files

        if deleted_count > 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Cleaned up {deleted_count} legacy text embedding caches for {filename_no_ext}")

    def get_caption(
            self: 'FileItemDTO',
            trigger=None,
            to_replace_list=None,
            add_if_not_present=False,
            short_caption=False
    ):
        if trigger is None and self.trigger_word is not None:
            trigger = self.trigger_word
        
        if trigger is not None and not self.is_reg:
            # add if not present if not regularization
            add_if_not_present = True
            
        if short_caption:
            raw_caption = self.raw_caption_short
        else:
            raw_caption = self.raw_caption
        if raw_caption is None:
            raw_caption = ''
        # handle dropout
        # Note: When cache_text_embeddings=True, caption dropout is intentionally
        # skipped here so that the full caption is always cached to disk.
        # Caption dropout for cached embeddings is applied at training time
        # in SDTrainer._apply_caption_dropout() by randomly replacing cached
        # per-image embeddings with blank embeddings.
        # For mixed I2V/T2V datasets, use the appropriate dropout rate based on mode.
        is_t2v_mode = getattr(self, 'is_i2v_mode', True) == False
        caption_dropout_rate = self.dataset_config.caption_dropout_rate_t2v if is_t2v_mode else self.dataset_config.caption_dropout_rate
        if caption_dropout_rate > 0 and not short_caption and not self.dataset_config.cache_text_embeddings:
            # get a random float form 0 to 1
            rand = random.random()
            if rand < caption_dropout_rate:
                # drop the caption
                return ''

        # get tokens
        token_list = raw_caption.split(',')

        # handle token dropout
        if self.dataset_config.token_dropout_rate > 0 and not short_caption and not self.dataset_config.cache_text_embeddings:
            new_token_list = []
            keep_tokens: int = self.dataset_config.keep_tokens
            for idx, token in enumerate(token_list):
                if idx < keep_tokens:
                    new_token_list.append(token)
                elif self.dataset_config.token_dropout_rate >= 1.0:
                    # drop the token
                    pass
                else:
                    # get a random float form 0 to 1
                    rand = random.random()
                    if rand > self.dataset_config.token_dropout_rate:
                        # keep the token
                        new_token_list.append(token)
            token_list = new_token_list

        if self.dataset_config.shuffle_tokens:
            random.shuffle(token_list)

        # join back together
        caption = ', '.join(token_list)
        caption = inject_trigger_into_prompt(caption, trigger, to_replace_list, add_if_not_present)

        if self.dataset_config.random_triggers:
            num_triggers = self.dataset_config.random_triggers_max
            if num_triggers > 1:
                num_triggers = random.randint(0, num_triggers)

            if num_triggers > 0:
                triggers = random.sample(self.dataset_config.random_triggers, num_triggers)
                caption = caption + ', ' + ', '.join(triggers)
                # add random triggers
                # for i in range(num_triggers):
                #     # fastest method
                #     trigger = self.dataset_config.random_triggers[int(random.random() * (len(self.dataset_config.random_triggers)))]
                #     caption = caption + ', ' + trigger

        if self.dataset_config.shuffle_tokens:
            # shuffle again
            token_list = caption.split(',')
            random.shuffle(token_list)
            caption = ', '.join(token_list)
        if caption == '':
            pass
        return caption

class AudioProcessingDTOMixin:
    def load_and_process_audio(self: 'FileItemDTO'):
        # Default to "no audio" unless we successfully extract it
        self.audio_data = None
        self.audio_tensor = None
        self.tensor = None
        _df = dataset_crypto.open_dataset(self.path)
        try:
            import torchaudio

            waveform, sample_rate = _df.open_audio()  # [channels, samples]
            waveform = waveform_to_stereo(waveform)  # Convert to stereo if not already
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
            self.tensor = waveform
            self.audio_tensor = waveform
            self.audio_data = {"waveform": waveform, "sample_rate": int(self.sample_rate)}

        except Exception as e:
            # if issue with libtorchcodec "Could not load libtorchcodec"
            raise Exception(f"** WARNING ** - Error Processing audio for {self.path}. Error: {e}")
        finally:
            _df.cleanup()
        

class ImageProcessingDTOMixin:
    def load_and_process_video(
        self: 'FileItemDTO',
        transform: Union[None, transforms.Compose],
        only_load_latents=False
    ):
        
        if self.augments is not None and len(self.augments) > 0:
            raise Exception('Augments not supported for videos')
            
        if self.has_augmentations:
            raise Exception('Augmentations not supported for videos')
        
        if not self.dataset_config.buckets:
            raise Exception('Buckets required for video processing')
        
        do_audio = self.dataset_config.do_audio
        
        _df = dataset_crypto.open_dataset(self.path)
        try:
            # Use OpenCV (plain) or PyAV (encrypted, in-RAM) to capture video frames
            cap = _df.open_video()
            
            if not cap.is_opened():
                raise Exception(f"Failed to open video file: {self.path}")
            
            # Get video properties
            total_frames = int(cap.total_frames)
            video_fps = cap.fps
            
            # Calculate the max valid frame index (accounting for zero-indexing)
            max_frame_index = total_frames - 1
            
            # Only log video properties if in debug mode
            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                print_acc(f"Video properties: {self.path}")
                print_acc(f"  Total frames: {total_frames}")
                print_acc(f"  Max valid frame index: {max_frame_index}")
                print_acc(f"  FPS: {video_fps}")
            
            frames_to_extract = []
            
            if self.dataset_config.auto_frame_count:
                # allow for any length video here but make sure it is temporally compressable.
                vid_length_seconds = total_frames / video_fps
                
                desired_num_frames = int(vid_length_seconds * self.dataset_config.fps)
                
                # make sure it is divisible by temporal_compression
                desired_num_frames = desired_num_frames // self.temporal_compression * self.temporal_compression
                
                # TODO, all models currently add a key frame, but future models may not, update here if this changes.
                desired_num_frames += 1  # add one for the key frame that is always added
                
                self.num_frames = desired_num_frames
                
            
            # Always stretch/shrink to the requested number of frames if needed
            if self.dataset_config.shrink_video_to_frames or total_frames < self.num_frames:
                # Distribute frames evenly across the entire video
                interval = max_frame_index / (self.num_frames - 1) if self.num_frames > 1 else 0
                frames_to_extract = [min(int(round(i * interval)), max_frame_index) for i in range(self.num_frames)]
            else:
                # Calculate frame interval based on FPS ratio
                fps_ratio = video_fps / self.dataset_config.fps
                frame_interval = max(1, int(round(fps_ratio)))
                
                # Calculate max consecutive frames we can extract at desired FPS
                max_consecutive_frames = (total_frames // frame_interval)
                
                if max_consecutive_frames < self.num_frames:
                    # Not enough frames at desired FPS, so stretch instead
                    interval = max_frame_index / (self.num_frames - 1) if self.num_frames > 1 else 0
                    frames_to_extract = [min(int(round(i * interval)), max_frame_index) for i in range(self.num_frames)]
                else:
                    # Calculate max start frame to ensure we can get all num_frames
                    max_start_frame = max_frame_index - ((self.num_frames - 1) * frame_interval)
                    start_frame = random.randint(0, max(0, max_start_frame))
                    
                    # Generate list of frames to extract
                    frames_to_extract = [start_frame + (i * frame_interval) for i in range(self.num_frames)]
                    
            # Final safety check - ensure no frame exceeds max valid index
            frames_to_extract = [min(frame_idx, max_frame_index) for frame_idx in frames_to_extract]
            
            # Only log frames to extract if in debug mode
            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                print_acc(f"  Frames to extract: {frames_to_extract}")
            
            # Extract frames
            frames = []
            for frame_idx in frames_to_extract:
                # Safety check - ensure frame_idx is within bounds (silently fix)
                if frame_idx > max_frame_index:
                    frame_idx = max_frame_index
                
                # Read the frame (RGB) - works for both plain (cv2) and encrypted (PyAV) sources
                frame = cap.read_frame(frame_idx)
                if frame is None:
                    # Try to read a nearby frame as a fallback
                    fallback_success = False
                    for fallback_offset in [1, -1, 5, -5, 10, -10]:
                        fallback_pos = max(0, min(frame_idx + fallback_offset, max_frame_index))
                        fallback_frame = cap.read_frame(fallback_pos)
                        if fallback_frame is not None:
                            # Only log in debug mode
                            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                                print_acc(f"Falling back to nearby frame {fallback_pos} instead of {frame_idx}")
                            frame = fallback_frame
                            fallback_success = True
                            break
                    if not fallback_success:
                        video_info = f"Video: {self.path}, Total frames: {total_frames}, FPS: {video_fps}"
                        raise Exception(f"Failed to read frame {frame_idx} from video. {video_info}")
                
                # Convert to PIL Image
                img = Image.fromarray(frame)
                
                # Apply the same processing as for single images
                img = img.convert('RGB')
                
                if self.flip_x:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if self.flip_y:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                
                # Apply bucketing
                img = img.resize((self.scale_to_width, self.scale_to_height), get_resize_method(self.dataset_config.resize_method))
                img = img.crop((
                    self.crop_x,
                    self.crop_y,
                    self.crop_x + self.crop_width,
                    self.crop_y + self.crop_height
                ))
                
                # Apply transform if provided
                if transform:
                    img = transform(img)
                
                frames.append(img)
            
            # Release the video capture
            cap.release()
            
            # Stack frames into tensor [frames, channels, height, width]
            self.tensor = torch.stack(frames)

            # ------------------------------
            # Audio extraction + stretching
            # ------------------------------
            if do_audio:
                # Default to "no audio" unless we successfully extract it
                self.audio_data = None
                self.audio_tensor = None

                try:
                    import torchaudio
                    import torch.nn.functional as F

                    # Compute the time range of the selected frames in the *source* video
                    # Include the last frame by extending to the next frame boundary.
                    if video_fps and video_fps > 0 and len(frames_to_extract) > 0:
                        clip_start_frame = int(frames_to_extract[0])
                        clip_end_frame = int(frames_to_extract[-1])
                        clip_start_time = clip_start_frame / float(video_fps)
                        clip_end_time = (clip_end_frame + 1) / float(video_fps)
                        source_duration = max(0.0, clip_end_time - clip_start_time)
                    else:
                        clip_start_time = 0.0
                        clip_end_time = 0.0
                        source_duration = 0.0

                    # Target duration is how this sampled/stretched clip is interpreted for training
                    # (i.e. num_frames at the configured dataset FPS).
                    if hasattr(self.dataset_config, "fps") and self.dataset_config.fps and self.dataset_config.fps > 0:
                        target_duration = float(self.num_frames) / float(self.dataset_config.fps)
                    else:
                        target_duration = source_duration

                    waveform, sample_rate = _df.open_audio()  # [channels, samples]
                    
                    waveform = waveform_to_stereo(waveform)  # Convert to stereo if not already
                    
                    if self.dataset_config.audio_normalize:
                        peak = waveform.abs().amax()  # global peak across channels
                        eps = 1e-9
                        target_peak = 0.999  # ~ -0.01 dBFS
                        gain = target_peak / (peak + eps)
                        waveform = waveform * gain

                    # Slice to the selected clip region (when we have a meaningful time range)
                    if source_duration > 0.0:
                        start_sample = int(round(clip_start_time * sample_rate))
                        end_sample = int(round(clip_end_time * sample_rate))
                        start_sample = max(0, min(start_sample, waveform.shape[-1]))
                        end_sample = max(0, min(end_sample, waveform.shape[-1]))
                        if end_sample > start_sample:
                            waveform = waveform[..., start_sample:end_sample]
                        else:
                            # No valid audio segment
                            waveform = None
                    else:
                        # If we can't compute a meaningful time range, treat as no-audio
                        waveform = None

                    if waveform is not None and waveform.numel() > 0:
                        target_samples = int(round(target_duration * sample_rate))
                        if target_samples > 0 and waveform.shape[-1] != target_samples:
                            # Time-stretch/shrink to match the video clip duration implied by dataset FPS.
                            if self.dataset_config.audio_preserve_pitch:
                                waveform = time_stretch_preserve_pitch(waveform, sample_rate, target_samples)  # waveform is [C, L]
                            else:
                                # Use linear interpolation over the time axis.
                                wf = waveform.unsqueeze(0)  # [1, C, L]
                                wf = F.interpolate(wf, size=target_samples, mode="linear", align_corners=False)
                                waveform = wf.squeeze(0)  # [C, L]

                        self.audio_tensor = waveform
                        self.audio_data = {"waveform": waveform, "sample_rate": int(sample_rate)}

                except Exception as e:
                    # if issue with libtorchcodec "Could not load libtorchcodec"
                    raise Exception(f"** WARNING ** - Error Processing audio for {self.path}. Error: {e}")
            
            # Only log success in debug mode
            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                print_acc(f"Successfully loaded video with {len(frames)} frames: {self.path}")
            _df.cleanup()
        
        except Exception as e:
            # Print full traceback
            traceback.print_exc()
            
            # Provide more context about the error
            error_msg = str(e)
            try:
                if 'Failed to read frame' in error_msg and cap is not None:
                    # Try to get more info about the video that failed
                    cap_status = "Opened" if cap.is_opened() else "Closed"
                    reported_total = cap.total_frames if cap.is_opened() else "Unknown"

                    print_acc(f"Video details when error occurred:")
                    print_acc(f"  Cap status: {cap_status}")
                    print_acc(f"  Reported total frames: {reported_total}")

                    # Close the cap if it's still open
                    cap.release()
            except Exception as debug_err:
                print_acc(f"Error during error diagnosis: {debug_err}")
            finally:
                _df.cleanup()
            
            print_acc(f"Error: {error_msg}")
            print_acc(f"Error loading video: {self.path}")
            
            # Re-raise with more detailed information
            raise Exception(f"Video loading error ({self.path}): {error_msg}") from e
        
    def load_and_process_image(
            self: 'FileItemDTO',
            transform: Union[None, transforms.Compose],
            only_load_latents=False,
            force_load_images=False
    ):
        # handle get_prompt_embedding
        if self.is_text_embedding_cached:
            self.load_prompt_embedding()
        # if we are caching latents, just do that
        if self.is_latent_cached:
            self.get_latent()
            # if load_image_when_caching_latents is set, we still need the raw image
            # tensor in addition to the cached latent, so fall through to load it below
            # force_load_images can be used to bypass this for special cases like flow caching
            if not self.dataset_config.load_image_when_caching_latents and not force_load_images:
                if self.has_control_image:
                    self.load_control_image()
                if self.has_inpaint_image:
                    self.load_inpaint_image()
                if self.has_clip_image:
                    self.load_clip_image()
                if self.has_mask_image:
                    self.load_mask_image()
                if self.has_unconditional:
                    self.load_unconditional_image()
                return
        if self.is_audio_model:
            self.load_and_process_audio()
            return
        if self.dataset_config.num_frames > 1 or self.dataset_config.auto_frame_count:
            self.load_and_process_video(transform, only_load_latents)
            return
        try:
            _df = dataset_crypto.open_dataset(self.path)
            img = _df.open_image()  # exif-transposed; decrypted in RAM if encrypted
            _df.cleanup()
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.path}")

        if self.use_alpha_as_mask:
            # we do this to make sure it does not replace the alpha with another color
            # we want the image just without the alpha channel
            np_img = np.array(img)
            # strip off alpha
            np_img = np_img[:, :, :3]
            img = Image.fromarray(np_img)

        img = img.convert('RGB')
        w, h = img.size
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            print_acc(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            print_acc(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        if self.dataset_config.buckets:
            # scale and crop based on file item
            img = img.resize((self.scale_to_width, self.scale_to_height), get_resize_method(self.dataset_config.resize_method))
            # crop to x_crop, y_crop, x_crop + crop_width, y_crop + crop_height
            if img.width < self.crop_x + self.crop_width or img.height < self.crop_y + self.crop_height:
                # todo look into this. This still happens sometimes
                print_acc('size mismatch')
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))

            # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
        else:
            # Downscale the source image first
            # TODO this is nto right
            img = img.resize(
                (int(img.size[0] * self.dataset_config.scale), int(img.size[1] * self.dataset_config.scale)),
                get_resize_method(self.dataset_config.resize_method))
            min_img_size = min(img.size)
            if self.dataset_config.random_crop:
                if self.dataset_config.random_scale and min_img_size > self.dataset_config.resolution:
                    if min_img_size < self.dataset_config.resolution:
                        print_acc(
                            f"Unexpected values: min_img_size={min_img_size}, self.resolution={self.dataset_config.resolution}, image file={self.path}")
                        scale_size = self.dataset_config.resolution
                    else:
                        scale_size = random.randint(self.dataset_config.resolution, int(min_img_size))
                    scaler = scale_size / min_img_size
                    scale_width = int((img.width + 5) * scaler)
                    scale_height = int((img.height + 5) * scaler)
                    img = img.resize((scale_width, scale_height), get_resize_method(self.dataset_config.resize_method))
                img = transforms.RandomCrop(self.dataset_config.resolution)(img)
            else:
                img = transforms.CenterCrop(min_img_size)(img)
                img = img.resize((self.dataset_config.resolution, self.dataset_config.resolution), get_resize_method(self.dataset_config.resize_method))

        if self.augments is not None and len(self.augments) > 0:
            # do augmentations
            for augment in self.augments:
                if augment in transforms_dict:
                    img = transforms_dict[augment](img)

        if self.has_augmentations:
            # augmentations handles transforms
            img = self.augment_image(img, transform=transform)
        elif transform:
            img = transform(img)

        self.tensor = img
        if not only_load_latents:
            if self.has_control_image:
                self.load_control_image()
            if self.has_inpaint_image:
                self.load_inpaint_image()
            if self.has_clip_image:
                self.load_clip_image()
            if self.has_mask_image:
                self.load_mask_image()
            if self.has_unconditional:
                self.load_unconditional_image()


class InpaintControlFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_inpaint_image = False
        self.inpaint_path: Union[str, None] = None
        self.inpaint_tensor: Union[torch.Tensor, None] = None
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        if dataset_config.inpaint_path is not None:
            # find the control image path
            inpaint_path = dataset_config.inpaint_path
            # we are using control images
            img_path = kwargs.get('path', None)
            img_inpaint_ext_list = ['.png', '.webp']
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]

            for ext in img_inpaint_ext_list:
                p = os.path.join(inpaint_path, file_name_no_ext + ext)
                if os.path.exists(p):
                    self.inpaint_path = p
                    self.has_inpaint_image = True
                    break
                
    def load_inpaint_image(self: 'FileItemDTO'):
        try:
            # image must have alpha channel for inpaint
            # (decrypted in RAM on the fly if the dataset is encrypted)
            img = dataset_crypto.open_image(self.inpaint_path)
            # make sure has aplha
            if img.mode != 'RGBA':
                return
        
            w, h = img.size
            if w > h and self.scale_to_width < self.scale_to_height:
                # throw error, they should match
                raise ValueError(
                    f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            elif h > w and self.scale_to_height < self.scale_to_width:
                # throw error, they should match
                raise ValueError(
                    f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

            if self.flip_x:
                # do a flip
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.flip_y:
                # do a flip
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

            if self.dataset_config.buckets:
                # scale and crop based on file item
                img = img.resize((self.scale_to_width, self.scale_to_height), get_resize_method(self.dataset_config.resize_method))
                # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
                # crop
                img = img.crop((
                    self.crop_x,
                    self.crop_y,
                    self.crop_x + self.crop_width,
                    self.crop_y + self.crop_height
                ))
            else:
                raise Exception("Inpaint images not supported for non-bucket datasets")
            
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            if self.aug_replay_spatial_transforms:
                tensor = self.augment_spatial_control(img, transform=transform)
            else:
                tensor = transform(img)
            
            # is 0 to 1 with alpha
            self.inpaint_tensor = tensor
        
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.inpaint_path}")

    
    def cleanup_inpaint(self: 'FileItemDTO'):
        self.inpaint_tensor = None
                

class ControlFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_control_image = False
        self.control_path: Union[str, List[str], None] = None
        self.control_tensor: Union[torch.Tensor, None] = None
        self.control_tensor_list: Union[List[torch.Tensor], None] = None
        sd = kwargs.get('sd', None)
        self.use_raw_control_images = sd is not None and sd.use_raw_control_images
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.full_size_control_images = False
        if dataset_config.control_path is not None:
            # find the control image path
            control_path_list = dataset_config.control_path
            if not isinstance(control_path_list, list):
                control_path_list = [control_path_list]
            self.full_size_control_images = dataset_config.full_size_control_images
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            
            found_control_images = []
            for control_path in control_path_list:
                for ext in img_ext_list:
                    if os.path.exists(os.path.join(control_path, file_name_no_ext + ext)):
                        found_control_images.append(os.path.join(control_path, file_name_no_ext + ext))
                        self.has_control_image = True
                        break
            self.control_path = found_control_images
            if len(self.control_path) == 0:
                self.control_path = None
            elif len(self.control_path) == 1:
                # only do one
                self.control_path = self.control_path[0]

        if dataset_config.control_from_same_folder:
            # assume we have them. We will pull them on load.
            self.full_size_control_images = dataset_config.full_size_control_images
            self.has_control_image = True

    def get_new_control_paths(self: 'FileItemDTO'):
        if self.dataset_config.control_from_same_folder:
            # randomly grab image paths from the same folder as if they came from control_path
            pool_folder = os.path.dirname(self.path)
            # find all images in the folder
            img_files = []
            for ext in img_ext_list:
                img_files += glob.glob(os.path.join(pool_folder, f'*{ext}'))
            # remove the current image if len is greater than 1
            if len(img_files) > 1:
                img_files.remove(self.path)
            num_controls = min(self.dataset_config.num_controls_from_same_folder, len(img_files))
            # randomly grab them
            return random.sample(img_files, num_controls)
        else:
            return self.control_path

    def load_control_image(self: 'FileItemDTO'):
        control_tensors = []
        control_path_list = self.get_new_control_paths()
        if not isinstance(control_path_list, list):
            control_path_list = [control_path_list]
        
        for control_path in control_path_list:
            try:
                # decrypted in RAM on the fly if the dataset is encrypted
                img = dataset_crypto.open_image(control_path)

                if img.mode in ("RGBA", "LA"):
                    # Create a background with the specified transparent color
                    transparent_color = tuple(self.dataset_config.control_transparent_color)
                    background = Image.new("RGB", img.size, transparent_color)
                    # Paste the image on top using its alpha channel as mask
                    background.paste(img, mask=img.getchannel("A"))
                    img = background
                else:
                    # Already no alpha channel
                    img = img.convert("RGB")
            except Exception as e:
                print_acc(f"Error: {e}")
                print_acc(f"Error loading image: {control_path}")
            
            if not self.full_size_control_images:
                # we just scale them to 512x512:
                w, h = img.size
                img = img.resize((512, 512), get_resize_method(self.dataset_config.resize_method))

            elif not self.use_raw_control_images:
                w, h = img.size
                if self.flip_x:
                    # do a flip
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if self.flip_y:
                    # do a flip
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)

                if self.dataset_config.buckets:
                    # scale and crop based on file item
                    img = img.resize((self.scale_to_width, self.scale_to_height), get_resize_method(self.dataset_config.resize_method))
                    # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
                    # crop
                    img = img.crop((
                        self.crop_x,
                        self.crop_y,
                        self.crop_x + self.crop_width,
                        self.crop_y + self.crop_height
                    ))
                else:
                    raise Exception("Control images not supported for non-bucket datasets")
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            if self.aug_replay_spatial_transforms:
                tensor = self.augment_spatial_control(img, transform=transform)
            else:
                tensor = transform(img)
            control_tensors.append(tensor)
            
        if len(control_tensors) == 0:
            self.control_tensor = None
        elif len(control_tensors) == 1:
            self.control_tensor = control_tensors[0]
        elif self.use_raw_control_images:
            # just send the list of tensors as their shapes wont match
            self.control_tensor_list = control_tensors
        else:
            self.control_tensor = torch.stack(control_tensors, dim=0)

    def cleanup_control(self: 'FileItemDTO'):
        self.control_tensor = None
        self.control_tensor_list = None


class ClipImageFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_clip_image = False
        self.clip_image_path: Union[str, None] = None
        self.clip_image_tensor: Union[torch.Tensor, None] = None
        self.clip_image_embeds: Union[dict, None] = None
        self.clip_image_embeds_unconditional: Union[dict, None] = None
        self.has_clip_augmentations = False
        self.clip_image_aug_transform: Union[None, A.Compose] = None
        self.clip_image_processor: Union[None, CLIPImageProcessor] = None
        self.clip_image_encoder_path: Union[str, None] = None
        self.is_caching_clip_vision_to_disk = False
        self.is_vision_clip_cached = False
        self.clip_vision_is_quad = False
        self.clip_vision_load_device = 'cpu'
        self.clip_vision_unconditional_paths: Union[List[str], None] = None
        self._clip_vision_embeddings_path: Union[str, None] = None
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        if dataset_config.clip_image_path is not None or dataset_config.clip_image_from_same_folder:
            # copy the clip image processor so the dataloader can do it
            sd = kwargs.get('sd', None)
            if hasattr(sd.adapter, 'clip_image_processor'):
                self.clip_image_processor = sd.adapter.clip_image_processor
        if dataset_config.clip_image_path is not None:
            # find the control image path
            clip_image_path = dataset_config.clip_image_path
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(clip_image_path, file_name_no_ext + ext)):
                    self.clip_image_path = os.path.join(clip_image_path, file_name_no_ext + ext)
                    self.has_clip_image = True
                    break
            self.build_clip_imag_augmentation_transform()
            
        if dataset_config.clip_image_from_same_folder:
            # assume we have one. We will pull it on load.
            self.has_clip_image = True
            self.build_clip_imag_augmentation_transform()

    def build_clip_imag_augmentation_transform(self: 'FileItemDTO'):
        if self.dataset_config.clip_image_augmentations is not None and len(self.dataset_config.clip_image_augmentations) > 0:
            self.has_clip_augmentations = True
            augmentations = [Augments(**aug) for aug in self.dataset_config.clip_image_augmentations]

            if self.dataset_config.clip_image_shuffle_augmentations:
                random.shuffle(augmentations)

            augmentation_list = []
            for aug in augmentations:
                # make sure method name is valid
                assert hasattr(A, aug.method_name), f"invalid augmentation method: {aug.method_name}"
                # get the method
                method = getattr(A, aug.method_name)
                # add the method to the list
                augmentation_list.append(method(**aug.params))

            self.clip_image_aug_transform = A.Compose(augmentation_list)

    def augment_clip_image(self: 'FileItemDTO', img: Image, transform: Union[None, transforms.Compose], ):
        if self.dataset_config.clip_image_shuffle_augmentations:
            self.build_clip_imag_augmentation_transform()

        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        if self.clip_vision_is_quad:
            # image is in a 2x2 gris. split, run augs, and recombine
            # split
            img1, img2 = np.hsplit(open_cv_image, 2)
            img1_1, img1_2 = np.vsplit(img1, 2)
            img2_1, img2_2 = np.vsplit(img2, 2)
            # apply augmentations
            img1_1 = self.clip_image_aug_transform(image=img1_1)["image"]
            img1_2 = self.clip_image_aug_transform(image=img1_2)["image"]
            img2_1 = self.clip_image_aug_transform(image=img2_1)["image"]
            img2_2 = self.clip_image_aug_transform(image=img2_2)["image"]
            # recombine
            augmented = np.vstack((np.hstack((img1_1, img1_2)), np.hstack((img2_1, img2_2))))

        else:
            # apply augmentations
            augmented = self.clip_image_aug_transform(image=open_cv_image)["image"]

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        augmented_tensor = transforms.ToTensor()(augmented) if transform is None else transform(augmented)

        return augmented_tensor

    def get_clip_vision_info_dict(self: 'FileItemDTO'):
        item = OrderedDict([
            ("image_encoder_path", self.clip_image_encoder_path),
            ("filename", os.path.basename(self.clip_image_path)),
            ("is_quad", self.clip_vision_is_quad)
        ])
        # when adding items, do it after so we dont change old latents
        if self.flip_x:
            item["flip_x"] = True
        if self.flip_y:
            item["flip_y"] = True
        return item
    def get_clip_vision_embeddings_path(self: 'FileItemDTO', recalculate=False):
        if self._clip_vision_embeddings_path is not None and not recalculate:
            return self._clip_vision_embeddings_path
        else:
            # we store latents in a folder in same path as image called _latent_cache
            img_dir = os.path.dirname(self.clip_image_path)
            latent_dir = os.path.join(img_dir, '_clip_vision_cache')
            hash_dict = self.get_clip_vision_info_dict()
            filename_no_ext = os.path.splitext(os.path.basename(self.clip_image_path))[0]
            # get base64 hash of md5 checksum of hash_dict
            hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
            hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
            hash_str = hash_str.replace('=', '')
            self._clip_vision_embeddings_path = os.path.join(latent_dir, f'{filename_no_ext}_{hash_str}.safetensors')

        return self._clip_vision_embeddings_path
    
    def get_new_clip_image_path(self: 'FileItemDTO'):
        if self.dataset_config.clip_image_from_same_folder:
            # randomly grab an image path from the same folder
            pool_folder = os.path.dirname(self.path)
            # find all images in the folder
            img_files = []
            for ext in img_ext_list:
                img_files += glob.glob(os.path.join(pool_folder, f'*{ext}'))
            # remove the current image if len is greater than 1
            if len(img_files) > 1:
                img_files.remove(self.path)
            # randomly grab one
            return random.choice(img_files)
        else:
            return self.clip_image_path

    def load_clip_image(self: 'FileItemDTO'):
        is_dynamic_size_and_aspect = isinstance(self.clip_image_processor, PixtralVisionImagePreprocessorCompatible) or \
                                    isinstance(self.clip_image_processor, SiglipImageProcessor)
        if self.clip_image_processor is None:
            is_dynamic_size_and_aspect = True # serving it raw
        if self.is_vision_clip_cached:
            self.clip_image_embeds = dataset_crypto.load_safetensors(self.get_clip_vision_embeddings_path())

            # get a random unconditional image
            if self.clip_vision_unconditional_paths is not None:
                unconditional_path = random.choice(self.clip_vision_unconditional_paths)
                self.clip_image_embeds_unconditional = dataset_crypto.load_safetensors(unconditional_path)

            return
        clip_image_path = self.get_new_clip_image_path()
        try:
            _cdf = dataset_crypto.open_dataset(clip_image_path)
            img = _cdf.open_image().convert('RGB')
            _cdf.cleanup()
        except Exception as e:
            # make a random noise image
            img = Image.new('RGB', (self.dataset_config.resolution, self.dataset_config.resolution))
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {clip_image_path}")

        img = img.convert('RGB')

        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
        if is_dynamic_size_and_aspect:
            pass  # let the image processor handle it
        elif img.width != img.height:
            min_size = min(img.width, img.height)
            if self.dataset_config.square_crop:
                # center crop to a square
                img = transforms.CenterCrop(min_size)(img)
            else:
                # image must be square. If it is not, we will resize/squish it so it is, that way we don't crop out data
                # resize to the smallest dimension
                img = img.resize((min_size, min_size), get_resize_method(self.dataset_config.resize_method))

        if self.has_clip_augmentations:
            self.clip_image_tensor = self.augment_clip_image(img, transform=None)
        else:
            self.clip_image_tensor = transforms.ToTensor()(img)

        # random crop
        # if self.dataset_config.clip_image_random_crop:
        #     # crop up to 20% on all sides. Keep is square
        #     crop_percent = random.randint(0, 20) / 100
        #     crop_width = int(self.clip_image_tensor.shape[2] * crop_percent)
        #     crop_height = int(self.clip_image_tensor.shape[1] * crop_percent)
        #     crop_left = random.randint(0, crop_width)
        #     crop_top = random.randint(0, crop_height)
        #     crop_right = self.clip_image_tensor.shape[2] - crop_width - crop_left
        #     crop_bottom = self.clip_image_tensor.shape[1] - crop_height - crop_top
        #     if len(self.clip_image_tensor.shape) == 3:
        #         self.clip_image_tensor = self.clip_image_tensor[:, crop_top:-crop_bottom, crop_left:-crop_right]
        #     elif len(self.clip_image_tensor.shape) == 4:
        #         self.clip_image_tensor = self.clip_image_tensor[:, :, crop_top:-crop_bottom, crop_left:-crop_right]

        if self.clip_image_processor is not None:
            # run it
            tensors_0_1 = self.clip_image_tensor.to(dtype=torch.float16)
            clip_out = self.clip_image_processor(
                images=tensors_0_1,
                return_tensors="pt",
                do_resize=True,
                do_rescale=False,
            ).pixel_values
            self.clip_image_tensor = clip_out.squeeze(0).clone().detach()

    def cleanup_clip_image(self: 'FileItemDTO'):
        self.clip_image_tensor = None
        self.clip_image_embeds = None




class AugmentationFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_augmentations = False
        self.unaugmented_tensor: Union[torch.Tensor, None] = None
        # self.augmentations: Union[None, List[Augments]] = None
        self.dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.aug_transform: Union[None, A.Compose] = None
        self.aug_replay_spatial_transforms = None
        self.build_augmentation_transform()

    def build_augmentation_transform(self: 'FileItemDTO'):
        if self.dataset_config.augmentations is not None and len(self.dataset_config.augmentations) > 0:
            self.has_augmentations = True
            augmentations = [Augments(**aug) for aug in self.dataset_config.augmentations]

            if self.dataset_config.shuffle_augmentations:
                random.shuffle(augmentations)

            augmentation_list = []
            for aug in augmentations:
                # make sure method name is valid
                assert hasattr(A, aug.method_name), f"invalid augmentation method: {aug.method_name}"
                # get the method
                method = getattr(A, aug.method_name)
                # add the method to the list
                augmentation_list.append(method(**aug.params))

            # add additional targets so we can augment the control image
            self.aug_transform = A.ReplayCompose(augmentation_list, additional_targets={'image2': 'image'})

    def augment_image(self: 'FileItemDTO', img: Image, transform: Union[None, transforms.Compose], ):

        # rebuild each time if shuffle
        if self.dataset_config.shuffle_augmentations:
            self.build_augmentation_transform()

        # save the original tensor
        self.unaugmented_tensor = transforms.ToTensor()(img) if transform is None else transform(img)

        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # apply augmentations
        transformed = self.aug_transform(image=open_cv_image)
        augmented = transformed["image"]

        # save just the spatial transforms for controls and masks
        augmented_params = transformed["replay"]
        spatial_transforms = ['Rotate', 'Flip', 'HorizontalFlip', 'VerticalFlip', 'Resize', 'Crop', 'RandomCrop',
                              'ElasticTransform', 'GridDistortion', 'OpticalDistortion']
        # only store the spatial transforms
        augmented_params['transforms'] = [t for t in augmented_params['transforms'] if t['__class_fullname__'].split('.')[-1] in spatial_transforms]

        if self.dataset_config.replay_transforms:
            self.aug_replay_spatial_transforms = augmented_params

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        augmented_tensor = transforms.ToTensor()(augmented) if transform is None else transform(augmented)

        return augmented_tensor

    # augment control images spatially consistent with transforms done to the main image
    def augment_spatial_control(self: 'FileItemDTO', img: Image, transform: Union[None, transforms.Compose] ):
        if self.aug_replay_spatial_transforms is None:
            # no transforms
            return transform(img)

        # save colorspace to convert back to
        colorspace = img.mode

        # convert to rgb
        img = img.convert('RGB')

        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # Replay transforms
        transformed = A.ReplayCompose.replay(self.aug_replay_spatial_transforms, image=open_cv_image)
        augmented = transformed["image"]

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        # convert back to original colorspace
        augmented = augmented.convert(colorspace)

        augmented_tensor = transforms.ToTensor()(augmented) if transform is None else transform(augmented)
        return augmented_tensor

    def cleanup_control(self: 'FileItemDTO'):
        self.unaugmented_tensor = None


class MaskFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_mask_image = False
        self.mask_path: Union[str, None] = None
        self.mask_tensor: Union[torch.Tensor, None] = None
        self.use_alpha_as_mask: bool = False
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.mask_min_value = dataset_config.mask_min_value
        if dataset_config.alpha_mask:
            self.use_alpha_as_mask = True
            self.mask_path = kwargs.get('path', None)
            self.has_mask_image = True
        elif dataset_config.mask_path is not None:
            # find the control image path
            mask_path = dataset_config.mask_path if dataset_config.mask_path is not None else dataset_config.alpha_mask
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(mask_path, file_name_no_ext + ext)):
                    self.mask_path = os.path.join(mask_path, file_name_no_ext + ext)
                    self.has_mask_image = True
                    break

    def load_mask_image(self: 'FileItemDTO'):
        try:
            # decrypted in RAM on the fly if the dataset is encrypted
            img = dataset_crypto.open_image(self.mask_path)
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.mask_path}")

        if self.use_alpha_as_mask:
            # pipeline expectws an rgb image so we need to put alpha in all channels
            np_img = np.array(img)
            np_img[:, :, :3] = np_img[:, :, 3:]

            np_img = np_img[:, :, :3]
            img = Image.fromarray(np_img)

        img = img.convert('RGB')
        if self.dataset_config.invert_mask:
            img = ImageOps.invert(img)
        w, h = img.size
        fix_size = False
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            print_acc(f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            fix_size = True
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            print_acc(f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            fix_size = True

        if fix_size:
            # swap all the sizes
            self.scale_to_width, self.scale_to_height = self.scale_to_height, self.scale_to_width
            self.crop_width, self.crop_height = self.crop_height, self.crop_width
            self.crop_x, self.crop_y = self.crop_y, self.crop_x




        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # randomly apply a blur up to 0.5% of the size of the min (width, height)
        min_size = min(img.width, img.height)
        blur_radius = int(min_size * random.random() * 0.005)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # make grayscale
        img = img.convert('L')

        if self.dataset_config.buckets:
            # scale and crop based on file item
            img = img.resize((self.scale_to_width, self.scale_to_height), get_resize_method(self.dataset_config.resize_method))
            # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
            # crop
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))
        else:
            raise Exception("Mask images not supported for non-bucket datasets")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.aug_replay_spatial_transforms:
            self.mask_tensor = self.augment_spatial_control(img, transform=transform)
        else:
            self.mask_tensor = transform(img)
        self.mask_tensor = value_map(self.mask_tensor, 0, 1.0, self.mask_min_value, 1.0)
        # convert to grayscale

    def cleanup_mask(self: 'FileItemDTO'):
        self.mask_tensor = None


class UnconditionalFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_unconditional = False
        self.unconditional_path: Union[str, None] = None
        self.unconditional_tensor: Union[torch.Tensor, None] = None
        self.unconditional_latent: Union[torch.Tensor, None] = None
        self.unconditional_transforms = self.dataloader_transforms
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)

        if dataset_config.unconditional_path is not None:
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(dataset_config.unconditional_path, file_name_no_ext + ext)):
                    self.unconditional_path = os.path.join(dataset_config.unconditional_path, file_name_no_ext + ext)
                    self.has_unconditional = True
                    break

    def load_unconditional_image(self: 'FileItemDTO'):
        try:
            # decrypted in RAM on the fly if the dataset is encrypted
            img = dataset_crypto.open_image(self.unconditional_path)
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.mask_path}")

        img = img.convert('RGB')
        w, h = img.size
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        if self.dataset_config.buckets:
            # scale and crop based on file item
            img = img.resize((self.scale_to_width, self.scale_to_height), get_resize_method(self.dataset_config.resize_method))
            # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
            # crop
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))
        else:
            raise Exception("Unconditional images are not supported for non-bucket datasets")

        if self.aug_replay_spatial_transforms:
            self.unconditional_tensor = self.augment_spatial_control(img, transform=self.unconditional_transforms)
        else:
            self.unconditional_tensor = self.unconditional_transforms(img)

    def cleanup_unconditional(self: 'FileItemDTO'):
        self.unconditional_tensor = None
        self.unconditional_latent = None

class ArgBreakMixin:
    # just stops super calls form hitting object
    def __init__(self, *args, **kwargs):
        pass


class LatentCachingFileItemDTOMixin:
    def __init__(self, *args, **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self._encoded_latent: Union[torch.Tensor, None] = None
        self._cached_first_frame_latent: Union[torch.Tensor, None] = None
        self._cached_audio_latent: Union[torch.Tensor, None] = None
        self._latent_path: Union[str, None] = None
        self.is_latent_cached = False
        self.is_caching_to_disk = False
        self.is_caching_to_memory = False
        self.latent_load_device = 'cpu'
        # todo, increment this if we change the latent format to invalidate cache
        self.latent_version = 1

    def get_latent_info_dict(self: 'FileItemDTO'):
        item = OrderedDict([
            ("filename", os.path.basename(self.path)),
            ("scale_to_width", self.scale_to_width),
            ("scale_to_height", self.scale_to_height),
            ("crop_x", self.crop_x),
            ("crop_y", self.crop_y),
            ("crop_width", self.crop_width),
            ("crop_height", self.crop_height),
            ("latent_space_version", self.latent_space_version),
            ("latent_version", self.latent_version),
        ])
        is_video = False
        # when adding items, do it after so we dont change old latents
        if self.flip_x:
            item["flip_x"] = True
        if self.flip_y:
            item["flip_y"] = True
        if self.dataset_config.auto_frame_count:
            # don't store num frames here as it is calculated dynamically
            item["auto_frame_count"] = True
            is_video = True
        elif self.dataset_config.num_frames > 1:
            item["num_frames"] = self.dataset_config.num_frames
            is_video = True
        if is_video and self.dataset_config.fps != 24:
            # only add fps if it deviates from the default
            item["fps"] = self.dataset_config.fps
        if is_video and self.dataset_config.do_i2v:
                item["do_i2v"] = True
        if is_video and self.dataset_config.do_t2v:
                item["do_t2v"] = True
        if is_video and self.dataset_config.do_audio:
            item["do_audio"] = True
            if self.dataset_config.audio_normalize:
                item["audio_normalize"] = True
            if self.dataset_config.audio_preserve_pitch:
                item["audio_preserve_pitch"] = True
        if self.is_audio_model:
            item["is_audio_model"] = True
            item["sample_rate"] = self.sample_rate
        return item

    def get_latent_path(self: 'FileItemDTO', recalculate=False):
        if self._latent_path is not None and not recalculate:
            return self._latent_path
        else:
            # we store latents in a folder in same path as image called _latent_cache
            img_dir = os.path.dirname(self.path)
            latent_dir = os.path.join(img_dir, '_latent_cache')
            hash_dict = self.get_latent_info_dict()
            filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]
            # get base64 hash of md5 checksum of hash_dict
            hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
            hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
            hash_str = hash_str.replace('=', '')
            self._latent_path = os.path.join(latent_dir, f'{filename_no_ext}_{hash_str}.safetensors')

        return self._latent_path

    def cleanup_latent(self):
        # Streaming: latents live on disk and are loaded on the fly
        # (decrypted in RAM if the dataset is encrypted). The per-item
        # tensors only exist while the item is inside the rotating prefetch
        # ring, so always release them here - nothing is kept in RAM/VRAM
        # across items.
        self._encoded_latent = None
        self._cached_first_frame_latent = None
        self._cached_audio_latent = None

    def get_latent(self, device=None):
        if not self.is_latent_cached:
            return None
        if self._encoded_latent is None:
            # load it from disk (decrypted in RAM if the dataset is encrypted)
            state_dict = dataset_crypto.load_safetensors(
                self.get_latent_path(),
                # device=device if device is not None else self.latent_load_device
                device='cpu'
            )
            self._encoded_latent = state_dict['latent']
            if 'first_frame_latent' in state_dict:
                self._cached_first_frame_latent = state_dict['first_frame_latent']
            if 'audio_latent' in state_dict:
                self._cached_audio_latent = state_dict['audio_latent']
            if 'num_frames' in state_dict:
                self.num_frames = int(state_dict['num_frames'].item())
        return self._encoded_latent


class LatentCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.latent_cache = {}

    def cache_latents_all_latents(self: 'AiToolkitDataset'):
        with accelerator.main_process_first():
            print_acc(f"Caching latents for {self.dataset_path}")
            # Streaming design: latents are always written to disk and are
            # loaded on the fly (decrypted in RAM if the dataset is
            # encrypted) by the prefetch pipeline during training. Nothing
            # is kept in RAM/VRAM after this pass.
            print_acc(" - Saving latents to disk (streamed from disk at training time)")
            # move sd items to cpu except for vae
            self.sd.set_device_state_preset('cache_latents')

            # use tqdm to show progress
            i = 0
            for file_item in tqdm(self.file_list, desc='Caching latents to disk'):
                file_item.is_caching_to_disk = True
                file_item.is_caching_to_memory = False
                file_item.latent_load_device = self.sd.device

                latent_path = file_item.get_latent_path(recalculate=True)
                # check if it is saved to disk already
                if os.path.exists(latent_path):
                    # already cached; it will be loaded on demand (on the
                    # fly, decrypted in RAM if encrypted) during training
                    pass
                else:
                    # not saved to disk, calculate
                    # load the image first
                    file_item.load_and_process_image(self.transform, only_load_latents=True)
                    # fp32 when the model's front-end runs in fp32 (TREAD fp32_front)
                    dtype = self.sd.get_cache_dtype()
                    device = self.sd.device_torch
                    state_dict = OrderedDict()
                    first_frame_latent = None
                    audio_latent = None
                    frames = None
                    # add batch dimension
                    try:
                        imgs = file_item.tensor.unsqueeze(0).to(device, dtype=dtype)
                        latent = self.sd.encode_images(imgs, dtype=dtype).squeeze(0)
                        state_dict['latent'] = latent.clone().detach().cpu()
                    except Exception as e:
                        print_acc(f"Error processing image: {file_item.path}")
                        print_acc(f"Error: {str(e)}")
                        raise e
                    # do first frame
                    is_video = self.dataset_config.auto_frame_count or self.dataset_config.num_frames > 1
                    if self.dataset_config.do_i2v:
                        frames = file_item.tensor.unsqueeze(0).to(device, dtype=dtype)
                        if len(frames.shape) == 4:
                            first_frames = frames
                        elif len(frames.shape) == 5:
                            first_frames = frames[:, 0]
                        else:
                            raise ValueError(f"Unknown frame shape {frames.shape}")
                        first_frame_latent = self.sd.encode_images(first_frames, dtype=dtype).squeeze(0)
                        state_dict['first_frame_latent'] = first_frame_latent.clone().detach().cpu()
                    
                    # audio (video+audio models only — audio-only models already encoded above via encode_images)
                    if not self.is_audio_model and file_item.audio_data is not None:
                        audio_latent = self.sd.encode_audio([file_item.audio_data]).squeeze(0)
                        state_dict['audio_latent'] = audio_latent.clone().detach().cpu()
                    
                    if is_video:
                        state_dict['num_frames'] = torch.tensor(file_item.num_frames, dtype=torch.int32)
                    
                    # save_latent (encrypted at rest when a dataset password is set).
                    # The tensors are NOT kept in memory - they are streamed
                    # back from disk (decrypted on the fly) during training.
                    meta = get_meta_for_safetensors(file_item.get_latent_info_dict())
                    dataset_crypto.save_safetensors(state_dict, latent_path, metadata=meta)

                    del imgs
                    del latent
                    del frames
                    del file_item.tensor
                    del state_dict
                    del first_frame_latent
                    del audio_latent
                    file_item.cleanup()

                file_item.is_latent_cached = True
                i += 1

            # restore device state
            self.sd.restore_device_state()


class TextEmbeddingFileItemDTOMixin:
    def __init__(self, *args, **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.prompt_embeds: Union[PromptEmbeds, None] = None
        self._text_embedding_path: Union[str, None] = None
        self.is_text_embedding_cached = False
        self.text_embedding_load_device = 'cpu'
        self.text_embedding_version = 1

    def get_text_embedding_info_dict(self: 'FileItemDTO', prompt_index: int = None):
        """
        Get info dict used for text embedding cache path generation.
        
        For JSON captions:
            - Uses JSON file hash instead of caption text for cache invalidation
            - prompt_index parameter identifies which prompt from the JSON file
            - This allows caching all prompts separately
        
        For .txt captions:
            - Uses caption text in hash (existing behavior)
        """
        # make sure the caption is loaded here
        if self.caption is None:
            self.load_caption()
        
        # For JSON captions, use JSON file hash instead of caption text
        # This allows caching all prompts from the same JSON file
        if self.json_caption_path is not None and self.json_file_hash is not None:
            item = OrderedDict([
                ("json_hash", self.json_file_hash),
                ("text_embedding_space_version", self.text_embedding_space_version),
                ("text_embedding_version", self.text_embedding_version),
            ])
            # Include prompt index for multi-prompt JSON files
            if prompt_index is not None:
                item["prompt_index"] = prompt_index
        else:
            # Standard behavior for .txt captions
            item = OrderedDict([
                ("caption", self.caption),
                ("text_embedding_space_version", self.text_embedding_space_version),
                ("text_embedding_version", self.text_embedding_version),
            ])
        
        # if we have a control image, cache the path
        if self.encode_control_in_text_embeddings and self.control_path is not None:
            item["control_path"] = self.control_path
        return item
    
    def get_prompt_cache_hash(self: 'FileItemDTO', prompt_text: str) -> str:
        """
        Compute the content-addressed cache key (sha256) for a prompt.

        The key is the sha256 of the exact text passed to the text encoder, so
        identical prompts - across dataset items or across JSON caption files -
        map to a single shared cache file.

        When control images are encoded into the text embedding, the control
        image path(s) are folded into the hash as well, since they contribute
        to the cached tensors and the same prompt with a different control
        image produces a different embedding.
        """
        hash_input = prompt_text
        if self.encode_control_in_text_embeddings and self.control_path is not None:
            control_path_list = self.control_path
            if not isinstance(control_path_list, list):
                control_path_list = [control_path_list]
            hash_input = hash_input + '\x00' + '\x00'.join(sorted(control_path_list))
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    def get_text_embedding_cache_dir(self: 'FileItemDTO') -> str:
        """
        Get the content-addressed text embedding cache directory.

        Embeddings are stored in ``_t_e_cache`` next to the media file, in a
        subdirectory keyed by the text encoder space/version so that caches
        from different text encoders never collide.
        """
        img_dir = os.path.dirname(self.path)
        te_dir = os.path.join(img_dir, '_t_e_cache')
        version_dir = f'{self.text_embedding_space_version}_v{self.text_embedding_version}'
        return os.path.join(te_dir, version_dir)

    def get_text_embedding_path_for_prompt(self: 'FileItemDTO', prompt_text: str) -> str:
        """
        Get the text embedding cache path for a prompt, named by the sha256
        of the prompt text.
        """
        cache_hash = self.get_prompt_cache_hash(prompt_text)
        return os.path.join(self.get_text_embedding_cache_dir(), f'{cache_hash}.safetensors')

    def get_json_prompt_cache_paths(self: 'FileItemDTO') -> List[str]:
        """
        Get cache paths for all prompts in a JSON caption file.
        Returns a list of paths, one per prompt (named by the prompt's sha256).
        """
        if self.json_caption_path is None or not self.raw_prompts:
            return []
        return [
            self.get_text_embedding_path_for_prompt(clean_caption(p['prompt']))
            for p in self.raw_prompts
        ]

    def get_captions_for_caching(self: 'FileItemDTO') -> List[str]:
        """
        Get all captions that need to be cached.
        For JSON captions: returns all prompts.
        For .txt captions: returns the single caption.
        """
        if self.json_caption_path is not None and self.raw_prompts:
            return [clean_caption(p['prompt']) for p in self.raw_prompts]
        else:
            return [self.caption] if self.caption else []

    def get_text_embedding_path(self: 'FileItemDTO', recalculate=False, prompt_index: int = None):
        """
        Get the text embedding cache path (content-addressed by prompt sha256).

        For JSON captions:
            - prompt_index identifies which prompt from the JSON file to use
            - the cache file is named by the sha256 of that prompt's text, so
              identical prompts across dataset items share one cache file

        For .txt captions:
            - prompt_index is ignored and the item's caption is used
        """
        if recalculate or self._text_embedding_path is None:
            if self.json_caption_path is not None and prompt_index is not None:
                # make sure the JSON prompts are loaded
                if not self.raw_prompts:
                    self.load_caption()
                if prompt_index < 0 or prompt_index >= len(self.raw_prompts):
                    raise Exception(
                        f"prompt_index {prompt_index} out of range for JSON captions "
                        f"{self.json_caption_path} ({len(self.raw_prompts)} prompts)"
                    )
                prompt_text = clean_caption(self.raw_prompts[prompt_index]['prompt'])
            else:
                # make sure the caption is loaded here
                if self.caption is None:
                    self.load_caption()
                prompt_text = self.caption
            self._text_embedding_path = self.get_text_embedding_path_for_prompt(prompt_text)

        return self._text_embedding_path

    def cleanup_text_embedding(self):
        if self.prompt_embeds is not None:
            # we are caching on disk, don't save in memory
            self.prompt_embeds = None

    def load_prompt_embedding(self, device=None):
        if not self.is_text_embedding_cached:
            return
        if self.prompt_embeds is None:
            # For JSON captions with multiple prompts, randomly select one at training time
            # This provides caption augmentation while still using cached embeddings
            if self.json_caption_path is not None and len(self.raw_prompts) > 0:
                # Filter prompts by training mode
                is_i2v = getattr(self, 'is_i2v_mode', True)
                filtered_prompts = _filter_prompts_by_mode(self.raw_prompts, is_i2v_mode=is_i2v, log_warning=False)
                
                if filtered_prompts:
                    # Select a prompt using weighted random selection, then look
                    # up its cache by the sha256 of the selected prompt's text.
                    selected = select_prompt_weighted(filtered_prompts)
                    prompt_text = clean_caption(selected['prompt'])
                    cache_path = self.get_text_embedding_path_for_prompt(prompt_text)
                    if os.path.exists(cache_path):
                        self.prompt_embeds = PromptEmbeds.load(cache_path)
                        # Update caption to match loaded embedding
                        self.selected_prompt = selected
                        self.raw_caption = selected['prompt']
                        self.raw_caption_short = self.raw_caption
                        # Recompute caption with trigger word etc.
                        self.caption = self.get_caption()
                        if self.raw_caption_short is not None:
                            self.caption_short = self.get_caption(short_caption=True)
                        return
                    # A prompt matched the current mode but its cache file is
                    # missing. The text encoder is unloaded while training with
                    # cached embeddings, so this cannot be re-encoded on the fly -
                    # fail loudly instead of silently training on an empty caption.
                    raise Exception(
                        f"Text embedding cache miss for prompt {prompt_text!r} "
                        f"(JSON captions: {self.json_caption_path}). Expected cache file "
                        f"{cache_path} does not exist. Re-run the job with text embedding "
                        f"caching enabled (or delete the dataset's _t_e_cache directory) "
                        f"so the cache is rebuilt."
                    )
                
                # No prompts matched the current training mode - use empty prompt.
                # Intentionally do NOT fallback to prompt_index=0, as that could
                # silently train on wrong-mode prompts (e.g., T2V-only dataset
                # getting I2V-only captions). This respects the JSON file as the
                # explicit source of truth.
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(
                    f"Did not find matching prompt - fallback to empty prompt "
                    f"(JSON captions: {self.json_caption_path}, "
                    f"mode={'I2V' if is_i2v else 'T2V'})"
                )
                return
            
            # Standard behavior for .txt captions only.
            # If json_caption_path is set but raw_prompts is empty, don't try to load
            # from a cache path that was never created (would cause FileNotFoundError).
            # This can happen when a JSON file is modified to have no valid prompts
            # after caches were created for the previous valid prompts.
            if self.json_caption_path is None:
                text_embedding_path = self.get_text_embedding_path()
                if not os.path.exists(text_embedding_path):
                    raise Exception(
                        f"Text embedding cache miss for caption {self.caption!r} "
                        f"(media file: {self.path}). Expected cache file "
                        f"{text_embedding_path} does not exist. Re-run the job with text "
                        f"embedding caching enabled (or delete the dataset's "
                        f"_t_e_cache directory) so the cache is rebuilt."
                    )
                self.prompt_embeds = PromptEmbeds.load(text_embedding_path)
            # else: JSON with no valid prompts - leave prompt_embeds as None (uses empty caption)

class TextEmbeddingCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.is_caching_text_embeddings = self.dataset_config.cache_text_embeddings

    def cache_text_embeddings(self: 'AiToolkitDataset'):
        with accelerator.main_process_first():
            print_acc(f"Caching text_embeddings for {self.dataset_path}")
            print_acc(" - Saving text embeddings to disk")
            
            did_move = False
            total_cached = 0
            # fp32 when the model's front-end runs in fp32 (TREAD fp32_front)
            cache_dtype = self.sd.get_cache_dtype()

            # use tqdm to show progress
            i = 0
            for file_item in tqdm(self.file_list, desc='Caching text embeddings to disk'):
                file_item.latent_load_device = self.sd.device
                # Load caption first so JSON captions are parsed (json_caption_path and raw_prompts set)
                file_item.load_caption()

                # For JSON captions, cache ALL prompts separately
                # For .txt captions, use existing single-caption behavior
                if file_item.json_caption_path is not None and file_item.raw_prompts:
                    # Cache each prompt from the JSON file
                    captions_to_cache = file_item.get_captions_for_caching()
                    
                    for prompt_idx, caption_text in enumerate(captions_to_cache):
                        text_embedding_path = file_item.get_text_embedding_path(recalculate=True, prompt_index=prompt_idx)
                        
                        # Skip if already cached
                        if os.path.exists(text_embedding_path):
                            continue
                        
                        # load if not loaded
                        if not did_move:
                            self.sd.set_device_state_preset('cache_text_encoder')
                            did_move = True
                            
                        # Encode prompt (control images are per-file, not per-prompt)
                        if file_item.encode_control_in_text_embeddings and file_item.control_path is not None:
                            ctrl_img_list = []
                            control_path_list = file_item.control_path
                            if not isinstance(file_item.control_path, list):
                                control_path_list = [control_path_list]
                            for ctrl_idx in range(len(control_path_list)):
                                try:
                                    # decrypted in RAM on the fly if encrypted
                                    img = dataset_crypto.open_image(control_path_list[ctrl_idx]).convert("RGB")
                                    # convert to 0 to 1 tensor
                                    img = (
                                        TF.to_tensor(img)
                                        .unsqueeze(0)
                                        .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                                    )
                                    ctrl_img_list.append(img)
                                except Exception as e:
                                    print_acc(f"Error: {e}")
                                    print_acc(f"Error loading control image: {control_path_list[ctrl_idx]}")
                            
                            if len(ctrl_img_list) == 0:
                                ctrl_img = None
                            elif not self.sd.has_multiple_control_images:
                                ctrl_img = ctrl_img_list[0]
                            else:
                                ctrl_img = ctrl_img_list
                            prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption_text, control_images=ctrl_img)
                        else:
                            prompt_embeds: PromptEmbeds = self.sd.encode_prompt(caption_text)

                        # save it (in fp32 when the front-end runs in fp32)
                        prompt_embeds = prompt_embeds.to(dtype=cache_dtype)
                        prompt_embeds.save(text_embedding_path)
                        del prompt_embeds
                        total_cached += 1
                else:
                    # Standard behavior for .txt captions
                    text_embedding_path = file_item.get_text_embedding_path(recalculate=True)
                    # only process if not saved to disk
                    if not os.path.exists(text_embedding_path):
                        # load if not loaded
                        if not did_move:
                            self.sd.set_device_state_preset('cache_text_encoder')
                            did_move = True
                            
                        if file_item.encode_control_in_text_embeddings and file_item.control_path is not None:
                            ctrl_img_list = []
                            control_path_list = file_item.control_path
                            if not isinstance(file_item.control_path, list):
                                control_path_list = [control_path_list]
                            for ctrl_idx in range(len(control_path_list)):
                                try:
                                    # decrypted in RAM on the fly if encrypted
                                    img = dataset_crypto.open_image(control_path_list[ctrl_idx]).convert("RGB")
                                    # convert to 0 to 1 tensor
                                    img = (
                                        TF.to_tensor(img)
                                        .unsqueeze(0)
                                        .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                                    )
                                    ctrl_img_list.append(img)
                                except Exception as e:
                                    print_acc(f"Error: {e}")
                                    print_acc(f"Error loading control image: {control_path_list[ctrl_idx]}")
                            
                            if len(ctrl_img_list) == 0:
                                ctrl_img = None
                            elif not self.sd.has_multiple_control_images:
                                ctrl_img = ctrl_img_list[0]
                            else:
                                ctrl_img = ctrl_img_list
                            prompt_embeds: PromptEmbeds = self.sd.encode_prompt(file_item.caption, control_images=ctrl_img)
                        else:
                            prompt_embeds: PromptEmbeds = self.sd.encode_prompt(file_item.caption)
                        # save it (in fp32 when the front-end runs in fp32)
                        prompt_embeds = prompt_embeds.to(dtype=cache_dtype)
                        prompt_embeds.save(text_embedding_path)
                        del prompt_embeds
                        total_cached += 1
                
                file_item.is_text_embedding_cached = True
                i += 1
            
            print_acc(f"Cached {total_cached} text embeddings to disk")
            # restore device state
            # if did_move:
            #     self.sd.restore_device_state()


class CLIPCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.clip_vision_num_unconditional_cache = 20
        self.clip_vision_unconditional_cache = []

    def cache_clip_vision_to_disk(self: 'AiToolkitDataset'):
        if not self.is_caching_clip_vision_to_disk:
            return
        with torch.no_grad():
            print_acc(f"Caching clip vision for {self.dataset_path}")

            print_acc(" - Saving clip to disk")
            # move sd items to cpu except for vae
            self.sd.set_device_state_preset('cache_clip')

            # make sure the adapter has attributes
            if self.sd.adapter is None:
                raise Exception("Error: must have an adapter to cache clip vision to disk")

            clip_image_processor: CLIPImageProcessor = None
            if hasattr(self.sd.adapter, 'clip_image_processor'):
                clip_image_processor = self.sd.adapter.clip_image_processor

            if clip_image_processor is None:
                raise Exception("Error: must have a clip image processor to cache clip vision to disk")

            vision_encoder: CLIPVisionModelWithProjection = None
            if hasattr(self.sd.adapter, 'image_encoder'):
                vision_encoder = self.sd.adapter.image_encoder
            if hasattr(self.sd.adapter, 'vision_encoder'):
                vision_encoder = self.sd.adapter.vision_encoder

            if vision_encoder is None:
                raise Exception("Error: must have a vision encoder to cache clip vision to disk")

            # move vision encoder to device
            vision_encoder.to(self.sd.device)

            is_quad = self.sd.adapter.config.quad_image
            image_encoder_path = self.sd.adapter.config.image_encoder_path

            dtype = self.sd.torch_dtype
            device = self.sd.device_torch
            if hasattr(self.sd.adapter, 'clip_noise_zero') and self.sd.adapter.clip_noise_zero:
                # just to do this, we did :)
                # need more samples as it is random noise
                self.clip_vision_num_unconditional_cache = self.clip_vision_num_unconditional_cache
            else:
                # only need one since it doesnt change
                self.clip_vision_num_unconditional_cache = 1

            # cache unconditionals
            print_acc(f" - Caching {self.clip_vision_num_unconditional_cache} unconditional clip vision to disk")
            clip_vision_cache_path = os.path.join(self.dataset_config.clip_image_path, '_clip_vision_cache')

            unconditional_paths = []

            is_noise_zero = hasattr(self.sd.adapter, 'clip_noise_zero') and self.sd.adapter.clip_noise_zero

            for i in range(self.clip_vision_num_unconditional_cache):
                hash_dict = OrderedDict([
                    ("image_encoder_path", image_encoder_path),
                    ("is_quad", is_quad),
                    ("is_noise_zero", is_noise_zero),
                ])
                # get base64 hash of md5 checksum of hash_dict
                hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
                hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
                hash_str = hash_str.replace('=', '')

                uncond_path = os.path.join(clip_vision_cache_path, f'uncond_{hash_str}_{i}.safetensors')
                if os.path.exists(uncond_path):
                    # skip it
                    unconditional_paths.append(uncond_path)
                    continue

                # generate a random image
                img_shape = (1, 3, self.sd.adapter.input_size, self.sd.adapter.input_size)
                if is_noise_zero:
                    tensors_0_1 = torch.rand(img_shape).to(device, dtype=torch.float32)
                else:
                    tensors_0_1 = torch.zeros(img_shape).to(device, dtype=torch.float32)
                clip_image = clip_image_processor(
                    images=tensors_0_1,
                    return_tensors="pt",
                    do_resize=True,
                    do_rescale=False,
                ).pixel_values

                if is_quad:
                    # split the 4x4 grid and stack on batch
                    ci1, ci2 = clip_image.chunk(2, dim=2)
                    ci1, ci3 = ci1.chunk(2, dim=3)
                    ci2, ci4 = ci2.chunk(2, dim=3)
                    clip_image = torch.cat([ci1, ci2, ci3, ci4], dim=0).detach()

                clip_output = vision_encoder(
                    clip_image.to(device, dtype=dtype),
                    output_hidden_states=True
                )
                # make state_dict ['last_hidden_state', 'image_embeds', 'penultimate_hidden_states']
                state_dict = OrderedDict([
                    ('image_embeds', clip_output.image_embeds.clone().detach().cpu()),
                    ('last_hidden_state', clip_output.hidden_states[-1].clone().detach().cpu()),
                    ('penultimate_hidden_states', clip_output.hidden_states[-2].clone().detach().cpu()),
                ])

                dataset_crypto.save_safetensors(state_dict, uncond_path)
                unconditional_paths.append(uncond_path)

            self.clip_vision_unconditional_cache = unconditional_paths

            # use tqdm to show progress
            i = 0
            for file_item in tqdm(self.file_list, desc=f'Caching clip vision to disk'):
                file_item.is_caching_clip_vision_to_disk = True
                file_item.clip_vision_load_device = self.sd.device
                file_item.clip_vision_is_quad = is_quad
                file_item.clip_image_encoder_path = image_encoder_path
                file_item.clip_vision_unconditional_paths = unconditional_paths
                if file_item.has_clip_augmentations:
                    raise Exception("Error: clip vision caching is not supported with clip augmentations")

                embedding_path = file_item.get_clip_vision_embeddings_path(recalculate=True)
                # check if it is saved to disk already
                if not os.path.exists(embedding_path):
                    # load the image first
                    file_item.load_clip_image()
                    # add batch dimension
                    clip_image = file_item.clip_image_tensor.unsqueeze(0).to(device, dtype=dtype)

                    if is_quad:
                        # split the 4x4 grid and stack on batch
                        ci1, ci2 = clip_image.chunk(2, dim=2)
                        ci1, ci3 = ci1.chunk(2, dim=3)
                        ci2, ci4 = ci2.chunk(2, dim=3)
                        clip_image = torch.cat([ci1, ci2, ci3, ci4], dim=0).detach()

                    clip_output = vision_encoder(
                        clip_image.to(device, dtype=dtype),
                        output_hidden_states=True
                    )

                    # make state_dict ['last_hidden_state', 'image_embeds', 'penultimate_hidden_states']
                    state_dict = OrderedDict([
                        ('image_embeds', clip_output.image_embeds.clone().detach().cpu()),
                        ('last_hidden_state', clip_output.hidden_states[-1].clone().detach().cpu()),
                        ('penultimate_hidden_states', clip_output.hidden_states[-2].clone().detach().cpu()),
                    ])
                    # metadata (encrypted at rest when a dataset password is set)
                    meta = get_meta_for_safetensors(file_item.get_clip_vision_info_dict())
                    dataset_crypto.save_safetensors(state_dict, embedding_path, metadata=meta)

                    del clip_image
                    del clip_output
                    del file_item.clip_image_tensor

                    # flush(garbage_collect=False)
                file_item.is_vision_clip_cached = True
                i += 1
            # flush every 100
            # if i % 100 == 0:
            #     flush()

        # restore device state
        self.sd.restore_device_state()



class ControlCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
            self.control_generator: ControlGenerator = None
    
    def add_control_path_to_file_item(self: 'AiToolkitDataset', file_item: 'FileItemDTO', control_path: str, control_type: ControlTypes):
        if control_type == 'inpaint':
            file_item.inpaint_path = control_path
            file_item.has_inpaint_image = True
        elif control_type == 'mask' or control_type == 'sapiens2_mask':
            file_item.mask_path = control_path
            file_item.has_mask_image = True
        else:
            if file_item.control_path is None:
                file_item.control_path = [control_path]
            elif isinstance(file_item.control_path, str):
                file_item.control_path = [file_item.control_path, control_path]
            elif isinstance(file_item.control_path, list):
                file_item.control_path.append(control_path)
            else:
                raise Exception(f"Error: control_path is not a string or list: {file_item.control_path}")
            file_item.has_control_image = True

    def setup_controls(self: 'AiToolkitDataset'):
        if not self.is_generating_controls:
            return
        with torch.no_grad():
            print_acc(f"Generating controls for {self.dataset_path}")
            device = self.sd.device
            
            self.control_generator = ControlGenerator(
                device=device,
                sd=self.sd,
            )

            # use tqdm to show progress
            for file_item in tqdm(self.file_list, desc=f'Generating Controls'):
                for control_type in self.dataset_config.controls:
                    # generates the control if it is not already there
                    control_path = self.control_generator.get_control_path(file_item.path, control_type)
                    if control_path is not None:
                        self.add_control_path_to_file_item(file_item, control_path, control_type)
                
            # remove models
            self.control_generator.cleanup()
            self.control_generator = None
            
            flush()
