"""
dataset.py

A new dataset module for loading Image-Text pairs from a JSONL file,
as requested for the Rep-to-Rep project.

This dataset is designed to replace the standard ImageFolder dataset
and provide (image_tensor, text_caption) pairs for text-to-image
conditional training.
"""

import os
import json
import numpy as np
from PIL import Image
import random
from typing import Callable, Iterator, Dict, List, Optional, Tuple
from torchvision.datasets import ImageFolder

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from glob import glob
import torch.distributed as dist

try:
    from huggingface_hub import HfFileSystem, get_token, hf_hub_url
except ImportError:
    print("huggingface_hub not installed")

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class IN1KTextJsonlDataset(Dataset):
    """
    A dataset that loads images and text captions from a JSONL file.

    The JSONL file is expected to have one JSON object per line,
    with at least two keys:
    - "path": Relative path to the image file (e.g., "n01440764/n01440764_9437.JPEG")
    - "recaption_short": The text caption for the image.
    """
    def __init__(self, 
                 jsonl_path: str, 
                 image_root: str, 
                 resolution: int = 256, 
                 transform: Optional[Callable] = None,
                 debug: bool = False,
                 debug_limit: int = 1000):
        """
        Args:
            jsonl_path (str): Path to the .jsonl file.
            image_root (str): Root directory of the image dataset (e.g., "path/to/imagenet").
            resolution (int): The target resolution for the images (e.g., 256).
            transform (Callable, optional): A custom transform function to apply to the PIL image.
                                           If None, a default transform is created.
            debug (bool): If True, only load a small subset of the data.
            debug_limit (int): Number of samples to load in debug mode.
        """
        super().__init__()
        self.image_root = image_root
        self.resolution = resolution
        
        # 1. Load and parse the JSONL file
        print(f"Loading JSONL from: {jsonl_path}")
        self.samples = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if debug and i >= debug_limit:
                        print(f"DEBUG mode: Loaded first {debug_limit} samples.")
                        break
                    try:
                        self.samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON line {i+1} in {jsonl_path}")
        except FileNotFoundError:
            print(f"ERROR: JSONL file not found at {jsonl_path}")
            raise
        
        if not self.samples:
            raise ValueError(f"No valid data loaded from {jsonl_path}.")

        # 2. Define the image transform
        if transform:
            self.transform = transform
        else:
            # Use the transform block specified by the user
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.resolution)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Add standard normalization for models working in [-1, 1] range
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns one item from the dataset.

        Returns:
            Tuple[torch.Tensor, str]:
                - image_tensor (C, H, W): The transformed image tensor, normalized to [-1, 1].
                - text_caption (str): The corresponding text caption.
        """
        # Get the metadata for the sample
        item = self.samples[idx]
        
        # 1. Load Text
        text_caption = item["recaption_short"]
        
        # 2. Load Image
        image_path = os.path.join(self.image_root, item["path"])
        
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except (IOError, OSError) as e:
            print(f"Warning: Could not load image {image_path}. Returning a dummy black image. Error: {e}")
            pil_image = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        
        # 3. Apply Transform
        image_tensor = self.transform(pil_image)
        
        return image_tensor, text_caption

class IN1KTextJsonlSampler(Dataset):
    """
    A dataset that loads images and text captions from a JSONL file.

    The JSONL file is expected to have one JSON object per line,
    with at least two keys:
    - "path": Relative path to the image file (e.g., "n01440764/n01440764_9437.JPEG")
    - "recaption_short": The text caption for the image.
    """
    def __init__(self, 
                 jsonl_path: str, 
                 debug: bool = False,
                 debug_limit: int = 1000):
        """
        Args:
            jsonl_path (str): Path to the .jsonl file.
            image_root (str): Root directory of the image dataset (e.g., "path/to/imagenet").
            resolution (int): The target resolution for the images (e.g., 256).
            transform (Callable, optional): A custom transform function to apply to the PIL image.
                                           If None, a default transform is created.
            debug (bool): If True, only load a small subset of the data.
            debug_limit (int): Number of samples to load in debug mode.
        """
        super().__init__()
        
        # 1. Load and parse the JSONL file
        print(f"Loading JSONL from: {jsonl_path}")
        self.samples = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if debug and i >= debug_limit:
                        print(f"DEBUG mode: Loaded first {debug_limit} samples.")
                        break
                    try:
                        self.samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON line {i+1} in {jsonl_path}")
        except FileNotFoundError:
            print(f"ERROR: JSONL file not found at {jsonl_path}")
            raise
        
        if not self.samples:
            raise ValueError(f"No valid data loaded from {jsonl_path}.")


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns one item from the dataset.

        Returns:
            Tuple[torch.Tensor, str]:
                - image_tensor (C, H, W): The transformed image tensor, normalized to [-1, 1].
                - text_caption (str): The corresponding text caption.
        """
        # Get the metadata for the sample
        item = self.samples[idx]
        
        # 1. Load Text
        text_caption = item["recaption_short"]
        id = item["id"]
        
        return id, text_caption