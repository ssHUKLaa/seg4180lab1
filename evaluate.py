"""Step 5 — Evaluation and Metrics.

Loads the best checkpoint, evaluates on the test set, and saves a
visualisation grid showing aerial image / ground-truth mask / predicted mask
for a sample of test images.

Usage:
    python evaluate.py
    python evaluate.py --samples 12 --threshold 0.5
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR = "dataset"
CKPT_PATH   = os.getenv("MODEL_CHECKPOINT", "checkpoints/best_model.pth")
PLOTS_DIR   = "plots"
IMG_SIZE    = 512
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
    model.load_state_dict(
        torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    print(f"Loaded checkpoint: {CKPT_PATH}")
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pred_bin, mask_bin):
    """Single-image IoU and Dice (numpy boolean arrays)."""
    intersection = (pred_bin & mask_bin).sum()
    union        = (pred_bin | mask_bin).sum()
    iou  = intersection / (union + 1e-7)
    dice = 2 * intersection / (pred_bin.sum() + mask_bin.sum() + 1e-7)
    return float(iou), float(dice)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",   type=int,   default=8,
                        help="Number of images to visualise (default: 8)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for binary mask (default: 0.5)")
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    # Collect test pairs
    img_dir  = os.path.join(DATASET_DIR, "test", "images")
    mask_dir = os.path.join(DATASET_DIR, "test", "masks")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    pairs = []
    for ip in img_paths:
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp   = os.path.join(mask_dir, stem + ".png")
        if os.path.exists(mp):
            pairs.append((ip, mp))

    print(f"Test images found: {len(pairs)}")
    model = load_model()

    all_iou  = []
    all_dice = []

    # Evaluate all test images
    print("Evaluating...")
    for ip, mp in pairs:
        img  = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")

        img_t = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(img_t)["out"]
            prob  = torch.sigmoid(logit)[0, 0].cpu().numpy()

        pred_bin = prob > args.threshold
        mask_np  = np.array(mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
        mask_bin = mask_np > 127

        iou, dice = compute_metrics(pred_bin, mask_bin)
        all_iou.append(iou)
        all_dice.append(dice)

    mean_iou  = np.mean(all_iou)
    mean_dice = np.mean(all_dice)
    print(f"\n=== Test Set Evaluation ===")
    print(f"Images evaluated : {len(pairs)}")
    print(f"Mean IoU         : {mean_iou:.4f}")
    print(f"Mean Dice        : {mean_dice:.4f}")

    # Save per-image CSV for reference
    csv_path = os.path.join(PLOTS_DIR, "test_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("filename,iou,dice\n")
        for (ip, _), iou, dice in zip(pairs, all_iou, all_dice):
            f.write(f"{os.path.basename(ip)},{iou:.4f},{dice:.4f}\n")
    print(f"Per-image metrics saved to {csv_path}")

    # ── Visualisation grid ────────────────────────────────────────────────────
    n = min(args.samples, len(pairs))
    # Pick evenly-spaced samples across the test set
    indices = np.linspace(0, len(pairs) - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["Aerial Image", "Ground-Truth Mask", "Predicted Mask"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=13, fontweight="bold")

    for row, idx in enumerate(indices):
        ip, mp = pairs[idx]
        img  = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")

        img_t = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(img_t)["out"]
            prob  = torch.sigmoid(logit)[0, 0].cpu().numpy()

        pred_mask = (prob > args.threshold).astype(np.uint8) * 255
        gt_mask   = np.array(mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))

        iou, dice = compute_metrics(pred_mask > 127, gt_mask > 127)

        # Denormalise image for display
        img_np = transform(img).numpy()
        img_np = (img_np * np.array(STD)[:, None, None] +
                  np.array(MEAN)[:, None, None]).clip(0, 1)
        img_np = img_np.transpose(1, 2, 0)

        axes[row][0].imshow(img_np)
        axes[row][1].imshow(gt_mask,   cmap="gray", vmin=0, vmax=255)
        axes[row][2].imshow(pred_mask, cmap="gray", vmin=0, vmax=255)

        for ax in axes[row]:
            ax.axis("off")
        axes[row][2].set_xlabel(f"IoU={iou:.3f}  Dice={dice:.3f}",
                                fontsize=9)
        axes[row][2].xaxis.set_visible(True)
        axes[row][2].tick_params(bottom=False, labelbottom=True)

    plt.suptitle(
        f"Test Set Predictions  —  Mean IoU: {mean_iou:.4f}  |  "
        f"Mean Dice: {mean_dice:.4f}",
        fontsize=14, y=1.005
    )
    plt.tight_layout()
    vis_path = os.path.join(PLOTS_DIR, "predictions.png")
    plt.savefig(vis_path, dpi=120, bbox_inches="tight")
    print(f"Visualisation saved to {vis_path}")
    df = pd.read_csv("plots/test_metrics.csv")
    print(df["iou"].describe())


if __name__ == "__main__":
    main()
