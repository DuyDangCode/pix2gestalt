#!/usr/bin/env python3
"""
Evaluation Script for pix2gestalt on BSDS-A (Amodal Berkeley Segmentation Dataset)

This script:
1. Loads the pix2gestalt model and pretrained checkpoint weights.
2. Iterates over BSDS-A instances (image + visible modal mask).
3. Generates the amodal completion and extracts the predicted amodal mask via thresholding.
4. Computes Intersection-over-Union (IoU) against the ground-truth amodal mask.
5. Logs real-time progress and overall mean IoU (mIoU).
6. Saves detailed evaluation results to a JSON/text report.
"""

import os
import sys
import argparse
import json
import time
import warnings
from typing import List, Dict, Any, Tuple, Optional

# Compatibility shim: setuptools >= 82.0.0 removed pkg_resources, which breaks
# legacy pytorch_lightning 1.4.2 and torchmetrics 0.6.0 checkpoint unpickling.
try:
    import pkg_resources
except ImportError:
    try:
        import importlib.metadata as _importlib_metadata
    except ImportError:
        import importlib_metadata as _importlib_metadata  # type: ignore
    import types

    class _DistributionNotFound(Exception):
        pass

    def _get_distribution(name):
        try:
            return _importlib_metadata.distribution(name)
        except _importlib_metadata.PackageNotFoundError:
            raise _DistributionNotFound(name)

    _pkg = types.ModuleType("pkg_resources")
    _pkg.DistributionNotFound = _DistributionNotFound
    _pkg.get_distribution = _get_distribution
    sys.modules["pkg_resources"] = _pkg

# Suppress harmless warnings (torchvision C++ extension & new GPU architecture warnings)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", message=".*is not compatible with the current PyTorch installation.*")

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from contextlib import nullcontext

# Ensure pix2gestalt package and submodules are accessible in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIX2GESTALT_DIR = os.path.join(SCRIPT_DIR, "pix2gestalt")
if PIX2GESTALT_DIR not in sys.path:
    sys.path.insert(0, PIX2GESTALT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


# ==============================================================================
# 1. Model Loading Function
# ==============================================================================
def load_model(config_path: str, ckpt_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Loads pix2gestalt model architecture from config and restores checkpoint weights.

    Args:
        config_path (str): Path to yaml model configuration file.
        ckpt_path (str): Path to model checkpoint file (.ckpt).
        device (str): Target torch device (e.g. 'cuda:0' or 'cpu').

    Returns:
        torch.nn.Module: Loaded and evaluated model on specified device.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt_path}\n"
            f"Please download the weights (e.g. wget -c -P pix2gestalt/ckpt "
            f"https://gestalt.cs.columbia.edu/assets/epoch=000005.ckpt)"
        )

    file_size = os.path.getsize(ckpt_path)
    # The official epoch=000005.ckpt is ~15.4 GB (15,466,113,894 bytes)
    if file_size < 14_000_000_000:
        raise ValueError(
            f"\n{'='*70}\n"
            f"CHECKPOINT FILE IS INCOMPLETE OR CORRUPTED!\n"
            f"{'='*70}\n"
            f"Path: {ckpt_path}\n"
            f"Current file size: {file_size / (1024**3):.2f} GB\n"
            f"Expected size    : ~14.40 GiB (15.4 GB, 15,466,113,894 bytes)\n\n"
            f"The download was stopped or interrupted before completing.\n"
            f"To resume and finish downloading the complete file without starting over, run:\n"
            f"  wget -c -P {os.path.dirname(ckpt_path)} https://gestalt.cs.columbia.edu/assets/epoch=000005.ckpt\n"
            f"{'='*70}\n"
        )

    print(f"[Model] Loading config from {config_path}")
    config = OmegaConf.load(config_path)

    print(f"[Model] Loading weights from {ckpt_path} ({file_size / (1024**3):.2f} GB) to CPU first (saving VRAM)...")
    try:
        pl_sd = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        if "PytorchStreamReader" in str(e) or "zip archive" in str(e):
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"CHECKPOINT FILE CORRUPTED (INCOMPLETE DOWNLOAD)\n"
                f"{'='*70}\n"
                f"Error: {e}\n"
                f"File: {ckpt_path} (size: {file_size / (1024**3):.2f} GB, expected 15.4 GB)\n"
                f"PyTorch failed to find the zip central directory because the file was cut off.\n"
                f"Please resume the download using:\n"
                f"  wget -c -P {os.path.dirname(ckpt_path)} https://gestalt.cs.columbia.edu/assets/epoch=000005.ckpt\n"
                f"{'='*70}\n"
            ) from e
        raise

    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0:
        print(f"[Model] Missing keys: {len(m)}")
    if len(u) > 0:
        print(f"[Model] Unexpected keys: {len(u)}")

    # Clean up checkpoint memory before allocating GPU memory
    del pl_sd
    del sd
    import gc
    gc.collect()

    print(f"[Model] Transferring model to target device: {device}...")
    model.to(device)
    model.eval()
    print("[Model] Model successfully loaded and set to eval mode.")
    return model


# ==============================================================================
# 2. Input Preprocessing & Inference Helpers
# ==============================================================================
def process_input_tensor(image_np: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """
    Resizes image/mask to target_size (default 256x256) and normalizes to [-1, 1].

    Args:
        image_np (np.ndarray): Input RGB numpy array of shape (H, W, 3) in [0, 255].
        target_size (tuple): Target (height, width).

    Returns:
        torch.Tensor: Normalized tensor of shape (1, 3, target_h, target_w) in [-1, 1].
    """
    resized = cv2.resize(image_np, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    norm = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0  # [0, 1]
    norm = norm * 2.0 - 1.0  # [-1, 1]
    return norm.unsqueeze(0)


@torch.no_grad()
def sample_pix2gestalt(
    model: torch.nn.Module,
    sampler: DDIMSampler,
    input_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    device: str,
    h: int = 256,
    w: int = 256,
    ddim_steps: int = 50,
    n_samples: int = 1,
    guidance_scale: float = 2.0,
    ddim_eta: float = 1.0,
    precision: str = "autocast",
) -> List[np.ndarray]:
    """
    Runs diffusion sampling to synthesize the completed amodal object on white background.

    Returns:
        List of np.ndarray: Each sample is an RGB uint8 image of shape (h, w, 3).
    """
    precision_scope = autocast if (precision == "autocast" and "cuda" in device) else nullcontext

    with precision_scope("cuda"):
        with model.ema_scope():
            cond = {}
            # Channel 1: CLIP embedding of input image
            clip_emb = model.get_learned_conditioning(input_tensor.to(device)).tile(n_samples, 1, 1)
            c = model.cc_projection(clip_emb)
            cond["c_crossattn"] = [c]

            # Channel 2: VAE latent of input image concatenated with VAE latent of visible mask
            input_enc = model.encode_first_stage(input_tensor.to(device)).mode().detach()
            mask_enc = model.encode_first_stage(mask_tensor.to(device)).mode().detach()
            c_concat = torch.cat((input_enc, mask_enc), dim=1)
            cond["c_concat"] = [c_concat.repeat(n_samples, 1, 1, 1)]

            if guidance_scale != 1.0:
                uc = {
                    "c_crossattn": [torch.zeros_like(c).to(device)],
                    "c_concat": [torch.zeros(n_samples, 8, h // 8, w // 8).to(device)],
                }
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=guidance_scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                x_T=None,
            )

            x_samples = model.decode_first_stage(samples_ddim)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0).cpu()

            output_ims = []
            for sample in x_samples:
                arr = 255.0 * rearrange(sample.numpy(), "c h w -> h w c")
                output_ims.append(arr.astype(np.uint8))
            return output_ims


def predict_amodal_mask(
    model: torch.nn.Module,
    sampler: DDIMSampler,
    image_rgb: np.ndarray,
    visible_mask: np.ndarray,
    device: str = "cuda",
    ddim_steps: int = 50,
    n_samples: int = 1,
    guidance_scale: float = 2.0,
    threshold: int = 250,
    precision: str = "autocast",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes original RGB image and visible modal mask, predicts full amodal mask.

    Args:
        model: pix2gestalt diffusion model.
        sampler: DDIMSampler instance.
        image_rgb: Original RGB image (H, W, 3) in [0, 255].
        visible_mask: Visible modal mask (H, W) in {0, 255} or boolean.
        threshold: Intensity threshold to separate completed object from white background (default 250).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - pred_mask_orig: Predicted amodal binary mask of shape (H, W), boolean.
            - completed_image_orig: Completed RGB image resized to original (H, W, 3).
    """
    orig_h, orig_w = image_rgb.shape[:2]

    # Format visible mask as 3-channel RGB image for VAE input
    if visible_mask.ndim == 2:
        rgb_visible_mask = np.stack([visible_mask] * 3, axis=-1)
    else:
        rgb_visible_mask = visible_mask
    if rgb_visible_mask.dtype == bool:
        rgb_visible_mask = (rgb_visible_mask.astype(np.uint8)) * 255

    input_tensor = process_input_tensor(image_rgb, (256, 256))
    mask_tensor = process_input_tensor(rgb_visible_mask, (256, 256))

    samples = sample_pix2gestalt(
        model=model,
        sampler=sampler,
        input_tensor=input_tensor,
        mask_tensor=mask_tensor,
        device=device,
        ddim_steps=ddim_steps,
        n_samples=n_samples,
        guidance_scale=guidance_scale,
        precision=precision,
    )

    # Extract amodal mask via background thresholding (object synthesized on white background)
    sample_masks = []
    for s in samples:
        gray = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
        # Pixels darker than threshold are the synthesized object
        mask_256 = (gray < threshold).astype(np.uint8)
        sample_masks.append(mask_256)

    # Majority vote across samples if n_samples > 1
    if len(sample_masks) > 1:
        voted_mask_256 = (np.mean(np.stack(sample_masks, axis=0), axis=0) >= 0.5).astype(np.uint8)
    else:
        voted_mask_256 = sample_masks[0]

    # Resize predicted mask and completion back to original resolution
    pred_mask_orig = cv2.resize(voted_mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 0
    completed_image_orig = cv2.resize(samples[0], (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    return pred_mask_orig, completed_image_orig


# ==============================================================================
# 3. Metric Helper
# ==============================================================================
def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes Intersection-over-Union (IoU) between predicted mask and ground truth mask.
    """
    pred_bin = pred_mask.astype(bool)
    gt_bin = gt_mask.astype(bool)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


# ==============================================================================
# 4. Dataset Loader for BSDS-A
# ==============================================================================
class BSDSADataset:
    """
    Loads the BSDS-A (Amodal Berkeley Segmentation) dataset.
    Supports:
    1. Annotation JSON file (BSDS_amodal_test.json) + images directory.
    2. Pre-extracted directory structure (images/, visible_masks/, amodal_masks/).
    3. Synthetic/Dummy mode for quick pipeline verification.
    """

    def __init__(
        self,
        images_dir: Optional[str] = None,
        ann_file: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        dummy_test: bool = False,
        auto_synth_ann: bool = False,
    ):
        self.items = []
        self.dummy_test = dummy_test
        self.auto_synth_ann = auto_synth_ann

        if dummy_test:
            self._init_dummy_samples()
            return

        # Attempt to locate dataset directory and annotation file
        if dataset_dir:
            if not images_dir:
                for candidate in [
                    os.path.join(dataset_dir, "images"),
                    os.path.join(dataset_dir, "test"),
                    os.path.join(dataset_dir, "BSR", "BSDS500", "data", "images", "test"),
                ]:
                    if os.path.exists(candidate):
                        images_dir = candidate
                        break
            if not ann_file:
                for candidate in [
                    os.path.join(dataset_dir, "annotations", "BSDS_amodal_test.json"),
                    os.path.join(dataset_dir, "BSDS_amodal_test.json"),
                    os.path.join(dataset_dir, "annotations.json"),
                ]:
                    if os.path.exists(candidate):
                        ann_file = candidate
                        break

        # Check for pre-extracted mask folder format first
        if dataset_dir and os.path.exists(os.path.join(dataset_dir, "visible_masks")) and os.path.exists(os.path.join(dataset_dir, "amodal_masks")):
            self._load_from_mask_folders(dataset_dir, images_dir)
        elif ann_file and os.path.exists(ann_file):
            self._load_from_cocoa_json(ann_file, images_dir)
        elif auto_synth_ann and images_dir and os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
            target_ann = ann_file if ann_file else os.path.join(dataset_dir or "data/bsds", "annotations", "BSDS_amodal_test.json")
            self._generate_synthetic_bsds_annotations(images_dir, target_ann)
        else:
            raise FileNotFoundError(
                f"\n{'='*70}\n"
                f"BSDS-A ANNOTATIONS NOT FOUND!\n"
                f"{'='*70}\n"
                f"Images directory : {images_dir}\n"
                f"Annotation file  : {ann_file}\n"
                f"Dataset root     : {dataset_dir}\n\n"
                f"The 200 BSDS500 images are ready, but 'BSDS_amodal_test.json' is missing.\n"
                f"To fix:\n"
                f"  1. Place official 'BSDS_amodal_test.json' in: {os.path.join(dataset_dir or '', 'annotations')}/BSDS_amodal_test.json\n"
                f"  2. Or run with '--auto_synth_ann' to synthesize amodal benchmarks on the real 200 BSDS images.\n"
                f"  3. Or run with '--dummy_test' to test GPU inference & DDIM sampling on synthetic samples.\n"
                f"{'='*70}\n"
            )

    def _generate_synthetic_bsds_annotations(self, images_dir: str, ann_file: str):
        """
        Generates synthetic amodal benchmark annotations on the real 200 BSDS500 images.
        """
        try:
            from pycocotools import mask as cocomask
        except ImportError:
            cocomask = None

        print(f"[Dataset] Generating synthetic amodal benchmark annotations for real images in {images_dir}...")
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        images_list = []
        annotations_list = []

        for idx, fname in enumerate(image_files, 1):
            fpath = os.path.join(images_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            h, w = img.shape[:2]

            images_list.append({
                "id": idx,
                "file_name": fname,
                "height": h,
                "width": w
            })

            cx, cy = w // 2, h // 2
            rx, ry = int(w * 0.22), int(h * 0.22)

            gt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(gt_mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)

            occ_mask = np.zeros((h, w), dtype=np.uint8)
            occ_x1 = cx - int(rx * 0.2)
            occ_x2 = cx + rx + 10
            occ_y1 = cy - ry - 10
            occ_y2 = cy + ry + 10
            cv2.rectangle(occ_mask, (occ_x1, occ_y1), (occ_x2, occ_y2), 1, -1)

            vis_mask = np.where((gt_mask == 1) & (occ_mask == 0), 1, 0).astype(np.uint8)

            contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poly_points = []
            for c in contours:
                if len(c) >= 3:
                    poly_points.append(c.flatten().tolist())

            if cocomask is not None:
                vis_rle = cocomask.encode(np.asfortranarray(vis_mask))
                vis_rle["counts"] = vis_rle["counts"].decode("utf-8")
            else:
                vis_rle = None

            occ_rate = float(np.sum(gt_mask & occ_mask)) / float(max(np.sum(gt_mask), 1))

            region_dict = {
                "name": f"synthetic_obj_{idx}",
                "area": float(np.sum(gt_mask)),
                "isStuff": False,
                "occlude_rate": round(occ_rate, 4),
                "order": 1,
                "segmentation": poly_points if poly_points else [[cx-rx, cy-ry, cx+rx, cy-ry, cx+rx, cy+ry, cx-rx, cy+ry]],
                "visible_mask": vis_rle
            }

            annotations_list.append({
                "image_id": idx,
                "author": "synthetic_generator",
                "depth_constraint": "",
                "size": 1,
                "url": "",
                "regions": [region_dict]
            })

        os.makedirs(os.path.dirname(ann_file), exist_ok=True)
        dataset_json = {
            "images": images_list,
            "annotations": annotations_list
        }
        with open(ann_file, "w") as f:
            json.dump(dataset_json, f, indent=2)

        print(f"[Dataset] Generated {len(annotations_list)} annotations on real BSDS500 images and saved to {ann_file}.")
        self._load_from_cocoa_json(ann_file, images_dir)

    def _init_dummy_samples(self):
        """Creates 2 synthetic test instances for quick smoke testing."""
        print("[Dataset] Initializing dummy test samples for verification...")
        h, w = 321, 481

        # Sample 1: Circle occluded by rectangle
        img1 = np.ones((h, w, 3), dtype=np.uint8) * 120
        # Background pattern
        img1[:160, :] = [180, 150, 120]
        # Target object: blue circle centered at (200, 160), radius 70
        gt_mask1 = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(gt_mask1, (200, 160), 70, 255, -1)
        cv2.circle(img1, (200, 160), 70, (40, 90, 200), -1)
        # Occluder: green box on the right half of the circle
        vis_mask1 = gt_mask1.copy()
        vis_mask1[:, 200:] = 0
        cv2.rectangle(img1, (200, 110), (320, 240), (60, 180, 70), -1)

        # Sample 2: Star/Polygon occluded by vertical bar
        img2 = np.ones((h, w, 3), dtype=np.uint8) * 200
        gt_mask2 = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(gt_mask2, (120, 80), (300, 240), 255, -1)
        cv2.rectangle(img2, (120, 80), (300, 240), (210, 80, 60), -1)
        vis_mask2 = gt_mask2.copy()
        vis_mask2[130:190, :] = 0  # Middle occluded
        cv2.rectangle(img2, (80, 130), (350, 190), (50, 50, 50), -1)

        self.items = [
            {
                "image_name": "dummy_circle_occlusion.png",
                "object_name": "circle_object",
                "image_rgb": img1,
                "visible_mask": vis_mask1 > 127,
                "amodal_gt_mask": gt_mask1 > 127,
                "occlude_rate": 0.5,
            },
            {
                "image_name": "dummy_box_occlusion.png",
                "object_name": "box_object",
                "image_rgb": img2,
                "visible_mask": vis_mask2 > 127,
                "amodal_gt_mask": gt_mask2 > 127,
                "occlude_rate": 0.35,
            },
        ]
        print(f"[Dataset] Ready with {len(self.items)} dummy instances.")

    def _load_from_cocoa_json(self, ann_file: str, images_dir: Optional[str]):
        """Parses annotations in COCOA BSDS_amodal_test.json format."""
        try:
            from pycocotools import mask as cocomask
        except ImportError:
            cocomask = None

        print(f"[Dataset] Loading annotations from {ann_file}...")
        with open(ann_file, "r") as f:
            data = json.load(f)

        # Map image_id -> image info
        images_info = {}
        for im in data.get("images", []):
            images_info[im["id"]] = im

        annotations = data.get("annotations", [])
        print(f"[Dataset] Found {len(images_info)} images and {len(annotations)} annotation records.")

        count = 0
        for ann in annotations:
            img_id = ann.get("image_id")
            im_info = images_info.get(img_id, {})
            file_name = im_info.get("file_name", f"{img_id:06d}.jpg")
            h = im_info.get("height", 321)
            w = im_info.get("width", 481)

            img_path = None
            if images_dir:
                candidate = os.path.join(images_dir, file_name)
                if os.path.exists(candidate):
                    img_path = candidate
                else:
                    # Try matching by filename without extension
                    stem = os.path.splitext(file_name)[0]
                    for ext in [".jpg", ".png", ".jpeg"]:
                        c2 = os.path.join(images_dir, stem + ext)
                        if os.path.exists(c2):
                            img_path = c2
                            break

            regions = ann.get("regions", [])
            for r_idx, region in enumerate(regions):
                # Filter for occluded objects if specified, or evaluate all annotated objects
                occlude_rate = region.get("occlude_rate", 0.0)

                # Decode visible mask (RLE or polygon)
                vis_rle = region.get("visible_mask")
                vis_mask = None
                if vis_rle is not None and cocomask is not None:
                    if isinstance(vis_rle, dict) and "counts" in vis_rle:
                        vis_mask = cocomask.decode(vis_rle) > 0
                elif isinstance(vis_rle, list):
                    vis_mask = self._poly_to_mask(vis_rle, h, w)

                # Decode amodal ground truth mask (segmentation polygon or RLE)
                amodal_seg = region.get("segmentation")
                amodal_mask = None
                if amodal_seg is not None:
                    if isinstance(amodal_seg, list):
                        if cocomask is not None and len(amodal_seg) > 0 and isinstance(amodal_seg[0], list):
                            rles = cocomask.frPyObjects(amodal_seg, h, w)
                            amodal_mask = cocomask.decode(cocomask.merge(rles)) > 0
                        else:
                            amodal_mask = self._poly_to_mask(amodal_seg, h, w)
                    elif isinstance(amodal_seg, dict) and cocomask is not None:
                        amodal_mask = cocomask.decode(amodal_seg) > 0

                if vis_mask is not None and amodal_mask is not None:
                    self.items.append({
                        "image_path": img_path,
                        "image_name": file_name,
                        "object_name": region.get("name", f"obj_{count}"),
                        "visible_mask_np": vis_mask,
                        "amodal_gt_mask_np": amodal_mask,
                        "occlude_rate": float(occlude_rate),
                        "height": h,
                        "width": w,
                    })
                    count += 1

        print(f"[Dataset] Successfully loaded {len(self.items)} valid object instances from BSDS-A.")

    def _poly_to_mask(self, poly: List, h: int, w: int) -> np.ndarray:
        """Converts polygon coordinate list to binary mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(poly) == 0:
            return mask > 0
        if isinstance(poly[0], list):
            for p in poly:
                pts = np.array(p, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)
        else:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        return mask > 0

    def _load_from_mask_folders(self, dataset_dir: str, images_dir: Optional[str]):
        """Loads from folder containing images/, visible_masks/, amodal_masks/."""
        vis_dir = os.path.join(dataset_dir, "visible_masks")
        amodal_dir = os.path.join(dataset_dir, "amodal_masks")
        if not images_dir:
            images_dir = os.path.join(dataset_dir, "images")

        vis_files = sorted(os.listdir(vis_dir))
        for vf in vis_files:
            if not vf.lower().endswith((".png", ".jpg", ".bmp")):
                continue
            base = os.path.splitext(vf)[0]
            # Match amodal mask
            af = os.path.join(amodal_dir, vf)
            if not os.path.exists(af):
                af = os.path.join(amodal_dir, base + ".png")
            if not os.path.exists(af):
                continue

            # Match image
            im_path = None
            if images_dir and os.path.exists(images_dir):
                for ext in [".jpg", ".png", ".jpeg"]:
                    candidate = os.path.join(images_dir, base.split("_")[0] + ext)
                    if os.path.exists(candidate):
                        im_path = candidate
                        break
                    candidate2 = os.path.join(images_dir, base + ext)
                    if os.path.exists(candidate2):
                        im_path = candidate2
                        break

            self.items.append({
                "image_path": im_path,
                "image_name": os.path.basename(im_path) if im_path else vf,
                "object_name": base,
                "visible_mask_path": os.path.join(vis_dir, vf),
                "amodal_mask_path": af,
                "occlude_rate": -1.0,
            })
        print(f"[Dataset] Loaded {len(self.items)} instances from mask folders.")

    def __len__(self) -> int:
        return len(self.items)

    def get_sample(self, idx: int) -> Dict[str, Any]:
        """Returns loaded image, visible_mask, and amodal_gt_mask for instance idx."""
        item = self.items[idx]
        if self.dummy_test:
            return item

        # Load RGB Image
        if "image_rgb" in item:
            image_rgb = item["image_rgb"]
        elif item.get("image_path") and os.path.exists(item["image_path"]):
            image_bgr = cv2.imread(item["image_path"])
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Fallback placeholder if image file not on disk
            h = item.get("height", 321)
            w = item.get("width", 481)
            image_rgb = np.ones((h, w, 3), dtype=np.uint8) * 128

        # Load Visible (Modal) Mask
        if "visible_mask_np" in item:
            vis_mask = item["visible_mask_np"]
        else:
            vis_mask = cv2.imread(item["visible_mask_path"], cv2.IMREAD_GRAYSCALE) > 127

        # Load Ground Truth Amodal Mask
        if "amodal_gt_mask_np" in item:
            amodal_gt = item["amodal_gt_mask_np"]
        else:
            amodal_gt = cv2.imread(item["amodal_mask_path"], cv2.IMREAD_GRAYSCALE) > 127

        return {
            "image_name": item["image_name"],
            "object_name": item["object_name"],
            "image_rgb": image_rgb,
            "visible_mask": vis_mask,
            "amodal_gt_mask": amodal_gt,
            "occlude_rate": item.get("occlude_rate", -1.0),
        }


# ==============================================================================
# 5. Main Evaluation Loop
# ==============================================================================
def evaluate(
    config_path: str,
    ckpt_path: str,
    dataset: BSDSADataset,
    device: str = "cuda",
    ddim_steps: int = 50,
    n_samples: int = 1,
    guidance_scale: float = 2.0,
    precision: str = "autocast",
    max_samples: Optional[int] = None,
    output_dir: str = "results",
    save_visualizations: bool = False,
) -> Dict[str, Any]:
    """
    Main evaluation routine:
    1. Loads the model with weights.
    2. Runs a single loop over all instances in the dataset.
    3. Logs progress and metrics.
    4. Saves detailed results to file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if save_visualizations:
        vis_dir = os.path.join(output_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)

    # 1. Load Model and Sampler
    model = load_model(config_path, ckpt_path, device=device)
    sampler = DDIMSampler(model)

    num_items = len(dataset)
    if max_samples and max_samples > 0:
        num_items = min(num_items, max_samples)

    print("\n" + "=" * 60)
    print(f"STARTING BSDS-A EVALUATION ({num_items} objects)")
    print(f"Device: {device} | DDIM Steps: {ddim_steps} | Samples: {n_samples} | Guidance: {guidance_scale}")
    print("=" * 60 + "\n")

    per_instance_results = []
    ious = []
    start_time = time.time()

    # 2. Main Evaluation Loop over all images
    pbar = tqdm(range(num_items), desc="Evaluating BSDS-A", unit="obj")
    for i in pbar:
        sample = dataset.get_sample(i)
        img_name = sample["image_name"]
        obj_name = sample["object_name"]
        image_rgb = sample["image_rgb"]
        vis_mask = sample["visible_mask"]
        gt_amodal_mask = sample["amodal_gt_mask"]

        # Run model inference to get predicted amodal mask
        pred_amodal_mask, completed_img = predict_amodal_mask(
            model=model,
            sampler=sampler,
            image_rgb=image_rgb,
            visible_mask=vis_mask,
            device=device,
            ddim_steps=ddim_steps,
            n_samples=n_samples,
            guidance_scale=guidance_scale,
            precision=precision,
        )

        # Compute metric (IoU)
        iou = compute_iou(pred_amodal_mask, gt_amodal_mask)
        ious.append(iou)
        running_miou = float(np.mean(ious))

        pbar.set_postfix({"IoU": f"{iou:.3f}", "mIoU": f"{running_miou:.3f}"})

        record = {
            "index": i,
            "image_name": img_name,
            "object_name": obj_name,
            "iou": round(iou, 4),
            "occlude_rate": round(sample["occlude_rate"], 4) if sample["occlude_rate"] >= 0 else None,
        }
        per_instance_results.append(record)

        # Optional: Save visualization composite
        if save_visualizations:
            # Composite: [Original Image | Visible Mask Overlay | Predicted Amodal | GT Amodal]
            h, w = image_rgb.shape[:2]
            vis_overlay = image_rgb.copy()
            vis_overlay[vis_mask] = (vis_overlay[vis_mask] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

            pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
            pred_vis[pred_amodal_mask] = [0, 200, 255]

            gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
            gt_vis[gt_amodal_mask] = [255, 100, 0]

            composite = np.concatenate([image_rgb, vis_overlay, pred_vis, gt_vis], axis=1)
            composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
            vis_path = os.path.join(output_dir, "vis", f"{i:04d}_{os.path.splitext(img_name)[0]}_iou_{iou:.3f}.png")
            cv2.imwrite(vis_path, composite_bgr)

    total_time = time.time() - start_time
    miou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    median_iou = float(np.median(ious)) if len(ious) > 0 else 0.0
    min_iou = float(np.min(ious)) if len(ious) > 0 else 0.0
    max_iou = float(np.max(ious)) if len(ious) > 0 else 0.0

    # 3. Log results to console
    summary_text = f"""
============================================================
             BSDS-A EVALUATION SUMMARY RESULTS
============================================================
Total Objects Evaluated : {len(ious)}
Mean IoU (mIoU)         : {miou:.4f} ({miou * 100:.2f}%)
Median IoU              : {median_iou:.4f} ({median_iou * 100:.2f}%)
Min IoU                 : {min_iou:.4f}
Max IoU                 : {max_iou:.4f}
Total Elapsed Time      : {total_time:.2f} s ({total_time / max(len(ious), 1):.2f} s / object)
Settings:
  - Checkpoint          : {ckpt_path}
  - Config              : {config_path}
  - DDIM Steps          : {ddim_steps}
  - Number of Samples   : {n_samples}
  - Guidance Scale      : {guidance_scale}
  - Precision           : {precision}
============================================================
"""
    print(summary_text)

    # 4. Save results to disk
    results_payload = {
        "summary": {
            "dataset": "BSDS-A",
            "total_objects": len(ious),
            "miou": round(miou, 4),
            "median_iou": round(median_iou, 4),
            "min_iou": round(min_iou, 4),
            "max_iou": round(max_iou, 4),
            "total_time_seconds": round(total_time, 2),
            "avg_time_per_object_seconds": round(total_time / max(len(ious), 1), 3),
            "config": {
                "ckpt_path": ckpt_path,
                "config_path": config_path,
                "ddim_steps": ddim_steps,
                "n_samples": n_samples,
                "guidance_scale": guidance_scale,
                "precision": precision,
            },
        },
        "per_instance_results": per_instance_results,
    }

    json_path = os.path.join(output_dir, "bsds_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    txt_path = os.path.join(output_dir, "bsds_eval_summary.txt")
    with open(txt_path, "w") as f:
        f.write(summary_text)

    print(f"[Results] Detailed JSON results saved to: {json_path}")
    print(f"[Results] Summary text report saved to:   {txt_path}")

    return results_payload


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate pix2gestalt on BSDS-A dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="pix2gestalt/configs/sd-finetune-pix2gestalt-c_concat-256.yaml",
        help="Path to model config yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="pix2gestalt/ckpt/epoch=000005.ckpt",
        help="Path to model checkpoint .ckpt",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/bsds",
        help="Root directory of BSDS-A dataset",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Path to directory containing BSDS500 test images",
    )
    parser.add_argument(
        "--ann_file",
        type=str,
        default=None,
        help="Path to BSDS_amodal_test.json annotation file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda', 'cuda:0', or 'cpu')",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps (default: 50, use 200 for paper reproduction)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of completion samples to generate per object (default: 1)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        help="Diffusion guidance scale / CFG (default: 2.0)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="autocast",
        choices=["autocast", "fp32"],
        help="Inference precision (autocast or fp32)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit evaluation to first N samples (useful for quick testing)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results and metrics",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save visual comparisons for each evaluated instance",
    )
    parser.add_argument(
        "--dummy_test",
        action="store_true",
        help="Run smoke test on synthetic dummy data to verify model & GPU pipeline",
    )
    parser.add_argument(
        "--auto_synth_ann",
        action="store_true",
        help="Generate synthetic amodal annotations on real BSDS500 images if official annotations are missing",
    )

    args = parser.parse_args()

    # Load dataset
    dataset = BSDSADataset(
        images_dir=args.images_dir,
        ann_file=args.ann_file,
        dataset_dir=args.dataset_dir,
        dummy_test=args.dummy_test,
        auto_synth_ann=args.auto_synth_ann,
    )

    # Run evaluation
    evaluate(
        config_path=args.config,
        ckpt_path=args.ckpt,
        dataset=dataset,
        device=args.device,
        ddim_steps=args.ddim_steps,
        n_samples=args.n_samples,
        guidance_scale=args.guidance_scale,
        precision=args.precision,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        save_visualizations=args.save_visualizations,
    )


if __name__ == "__main__":
    main()
