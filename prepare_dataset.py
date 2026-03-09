"""
Dataset preparation for house segmentation (Lab 2 - Step 3).

Approach (from Week 7 notebook):
  1. Load the keremberke/satellite-building-segmentation HuggingFace dataset.
  2. For each image, run SAM's automatic mask generator to get many candidate
     pixel-level masks.
  3. Match each SAM mask against the dataset's ground-truth building bounding
     boxes using IoU.  Any SAM mask with IoU > IOU_THRESHOLD against a labelled
     box is considered a building mask.
  4. OR all matched masks together into one binary mask per image.
  5. Save (image, mask) pairs to  dataset/{train,val,test}/{images,masks}/.

Requirements:
  - SAM checkpoint file (sam_vit_h_4b8939.pth, ~2.4 GB) in the project root,
    or set SAM_CHECKPOINT in your .env to point elsewhere.
  - Download command:
      curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o sam_vit_b_01ec64.pth
"""

import os
import sys
import threading
import numpy as np
from PIL import Image

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# ── Pause / quit control ──────────────────────────────────────────────────────
_paused = threading.Event()   # set   = paused,  clear = running
_quit   = threading.Event()   # set   = quit requested

def _input_listener():
    """Runs in a daemon thread. Reads commands from stdin."""
    print("Controls: type 'pause', 'resume', or 'quit' + Enter.")
    while not _quit.is_set():
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        if cmd == "pause":
            if _paused.is_set():
                print("[Already paused — type 'resume' to continue]")
            else:
                _paused.set()
                print("[Paused — type 'resume' to continue]")
        elif cmd == "resume":
            if not _paused.is_set():
                print("[Already running]")
            else:
                _paused.clear()
                print("[Resumed]")
        elif cmd == "quit":
            _quit.set()
            _paused.clear()   # unblock main loop so it exits cleanly
            print("[Quit requested — finishing current image then stopping]")
        else:
            print("Unknown command. Use: pause / resume / quit")

def _check_pause():
    """Blocks while paused. Returns False if quit was requested."""
    while _paused.is_set():
        threading.Event().wait(0.5)
    return not _quit.is_set()

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"
IOU_THRESHOLD = 0.3
OUTPUT_DIR = "dataset"
HF_TOKEN = os.getenv("HF_TOKEN") or None
# 'mini' works from cache; switch to 'full' if the Hub is accessible
DATASET_CONFIG = os.getenv("DATASET_CONFIG", "mini")

# Maps HuggingFace split names to output folder names
SPLITS = {"train": "train", "validation": "val", "test": "test"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_label_mask(bbox, img_w, img_h):
    """
    Convert a bounding box [x_min, y_min, width, height] to a binary (H, W)
    mask — same logic as the Week 7 notebook's make_mask(), but without the
    confusing transpose dance.
    """
    x_min, y_min, w, h = (int(v) for v in bbox)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y_min: y_min + h, x_min: x_min + w] = 1
    return mask


def build_pixel_mask(sam_masks, label_bboxes, img_w, img_h):
    """
    For every SAM-generated mask, check whether it overlaps sufficiently
    (IoU > IOU_THRESHOLD) with any labelled building bbox.  OR all matched
    masks together into one binary mask.

    Returns a uint8 array of shape (H, W) with values 0 or 255.
    """
    combined = np.zeros((img_h, img_w), dtype=np.uint8)

    for sam_box in sam_masks:
        sam_seg = sam_box["segmentation"].astype(np.uint8)  # bool -> 0/1

        for bbox in label_bboxes:
            label_seg = make_label_mask(bbox, img_w, img_h)
            union = np.sum(np.logical_or(sam_seg, label_seg))
            if union == 0:
                continue
            iou = np.sum(np.logical_and(sam_seg, label_seg)) / union
            if iou > IOU_THRESHOLD:
                combined = np.logical_or(combined, sam_seg).astype(np.uint8)
                break  # no need to check remaining label boxes for this mask

    return (combined * 255).astype(np.uint8)


def process_split(split_data, mask_generator, split_name):
    images_dir = os.path.join(OUTPUT_DIR, split_name, "images")
    masks_dir = os.path.join(OUTPUT_DIR, split_name, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    saved = 0
    skipped = 0
    for idx, example in enumerate(split_data):
        if not _check_pause():
            print(f"  [{split_name}] Stopped at index {idx}. {saved} saved, {skipped} skipped (already existed).")
            return False

        stem = f"{idx:05d}"
        img_path  = os.path.join(images_dir, f"{stem}.jpg")
        mask_path = os.path.join(masks_dir,  f"{stem}.png")

        # Resume-from-checkpoint: skip files already on disk
        if os.path.exists(img_path) and os.path.exists(mask_path):
            skipped += 1
            continue

        img = example["image"].convert("RGB")
        img_w, img_h = img.size
        bboxes = example["objects"]["bbox"]

        # Skip images with no building labels
        if not bboxes:
            continue

        img_array = np.array(img)
        sam_masks = mask_generator.generate(img_array)
        pixel_mask = build_pixel_mask(sam_masks, bboxes, img_w, img_h)

        img.save(img_path)
        Image.fromarray(pixel_mask).save(mask_path)
        saved += 1

        if (saved + skipped) % 10 == 0:
            print(f"  [{split_name}] {saved} saved, {skipped} skipped...")

    print(f"  [{split_name}] Done — {saved} new pairs saved, {skipped} already existed.")
    return True   # completed normally


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"ERROR: SAM checkpoint not found at '{SAM_CHECKPOINT}'.")
        print("Download it first with:")
        print(
            "  curl -L https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_h_4b8939.pth -o sam_vit_h_4b8939.pth"
        )
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: Running SAM on CPU is very slow.  A GPU is strongly recommended.")

    # Start the pause/quit listener in a background thread
    listener = threading.Thread(target=_input_listener, daemon=True)
    listener.start()

    print("Loading SAM model...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    # Reduce points_per_side and points_per_batch to fit in 4 GB VRAM.
    # Default points_per_side=32 and points_per_batch=64 OOM on small GPUs.
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        points_per_batch=16,
    )

    print(f"Loading dataset from HuggingFace (config='{DATASET_CONFIG}')...")
    ds = load_dataset(
        "keremberke/satellite-building-segmentation",
        name=DATASET_CONFIG,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    for hf_split, folder_name in SPLITS.items():
        if _quit.is_set():
            break
        if hf_split not in ds:
            print(f"Split '{hf_split}' not in dataset, skipping.")
            continue
        n = len(ds[hf_split])
        print(f"\nProcessing '{hf_split}' split ({n} images) -> {folder_name}/")
        completed = process_split(ds[hf_split], mask_generator, folder_name)
        if not completed:
            print("Stopped early — progress saved so far is in dataset/.")
            return

    print(f"\nDataset ready at '{OUTPUT_DIR}/'")
    print("Structure:")
    print("  dataset/train/images/  dataset/train/masks/")
    print("  dataset/val/images/    dataset/val/masks/")
    print("  dataset/test/images/   dataset/test/masks/")


if __name__ == "__main__":
    main()
