"""Training script for house segmentation (Lab 2 - Step 4).

Model:   DeepLabV3+ with ResNet-50 backbone, pretrained on COCO, fine-tuned.
Loss:    BCEWithLogitsLoss  (binary: house vs background)
Metrics: IoU, Dice score
Output:  checkpoints/best_model.pth   — best validation-IoU checkpoint
         checkpoints/last_model.pth   — weights after final epoch
         plots/training_curves.png    — loss / IoU / Dice curves

Usage:
    python train.py
    python train.py --epochs 30 --batch-size 4 --lr 5e-5
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# ── Config (overridable via CLI args) ─────────────────────────────────────────
DATASET_DIR = "dataset"
CKPT_DIR    = "checkpoints"
PLOTS_DIR   = "plots"
IMG_SIZE    = 512
BATCH_SIZE  = 2
EPOCHS      = 20
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalisation — backbone was pretrained on ImageNet
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── Dataset ───────────────────────────────────────────────────────────────────
class HouseSegDataset(Dataset):
    def __init__(self, split, img_size=IMG_SIZE, augment=False):
        self.augment  = augment
        self.img_size = img_size

        img_dir  = os.path.join(DATASET_DIR, split, "images")
        mask_dir = os.path.join(DATASET_DIR, split, "masks")

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.pairs = []
        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            mp   = os.path.join(mask_dir, stem + ".png")
            if os.path.exists(mp):
                self.pairs.append((ip, mp))

        self.img_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            if torch.rand(1) > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            if torch.rand(1) > 0.5:
                img  = TF.vflip(img)
                mask = TF.vflip(mask)

        img_t  = self.img_tf(img)
        # Mask values are 0 or 255 — normalise to 0.0 / 1.0
        mask_t = torch.from_numpy(np.array(mask)).float() / 255.0
        mask_t = mask_t.unsqueeze(0)  # (1, H, W)
        return img_t, mask_t


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    """DeepLabV3+ ResNet-50, COCO pretrained, classifier head replaced for
    binary (house / background) segmentation."""
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    # Replace the final 1×1 conv — COCO has 21 classes, we need 1
    model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pred_logits, masks):
    """Mean IoU and Dice score over a batch (after sigmoid + 0.5 threshold)."""
    preds        = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union        = (preds + masks).clamp(0, 1).sum(dim=(1, 2, 3))
    sum_pred     = preds.sum(dim=(1, 2, 3))
    sum_true     = masks.sum(dim=(1, 2, 3))
    iou  = (intersection / (union + 1e-7)).mean().item()
    dice = (2 * intersection / (sum_pred + sum_true + 1e-7)).mean().item()
    return iou, dice


# ── Epoch loop ────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training, epoch=None, total_epochs=None):
    model.train() if training else model.eval()
    total_loss = total_iou = total_dice = 0.0
    phase = "Train" if training else "Val  "
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)  # print ~10 updates per epoch

    for batch_idx, (imgs, masks) in enumerate(loader, 1):
        imgs, masks = imgs.to(device), masks.to(device)

        with torch.set_grad_enabled(training):
            outputs   = model(imgs)
            main_loss = criterion(outputs["out"], masks)
            loss      = main_loss
            # Auxiliary loss from the intermediate feature map (weighted 0.4)
            if "aux" in outputs and training:
                loss = loss + 0.4 * criterion(outputs["aux"], masks)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        iou, dice = compute_metrics(outputs["out"].detach(), masks)
        total_loss += main_loss.item()
        total_iou  += iou
        total_dice += dice

        if batch_idx % log_every == 0 or batch_idx == n_batches:
            epoch_str = f"Ep {epoch}/{total_epochs}  " if epoch else ""
            print(f"  {epoch_str}{phase} [{batch_idx:4d}/{n_batches}]  "
                  f"loss={total_loss/batch_idx:.4f}  "
                  f"iou={total_iou/batch_idx:.4f}",
                  flush=True)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train house segmentation model")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--img-size",   type=int,   default=IMG_SIZE)
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")

    train_ds = HouseSegDataset("train", img_size=args.img_size, augment=True)
    val_ds   = HouseSegDataset("val",   img_size=args.img_size)
    test_ds  = HouseSegDataset("test",  img_size=args.img_size)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # num_workers=0 avoids multiprocessing edge cases on Windows
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    model     = build_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history  = {k: [] for k in
                ["train_loss", "val_loss", "train_iou", "val_iou",
                 "train_dice", "val_dice"]}
    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_iou, tr_dice = run_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            training=True, epoch=epoch, total_epochs=args.epochs)
        vl_loss, vl_iou, vl_dice = run_epoch(
            model, val_loader, criterion, optimizer, DEVICE,
            training=False, epoch=epoch, total_epochs=args.epochs)

        scheduler.step(vl_iou)

        for key, val in zip(
            ["train_loss", "val_loss", "train_iou", "val_iou",
             "train_dice", "val_dice"],
            [tr_loss, vl_loss, tr_iou, vl_iou, tr_dice, vl_dice]
        ):
            history[key].append(val)

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f}  "
              f"IoU {tr_iou:.4f}/{vl_iou:.4f}  "
              f"Dice {tr_dice:.4f}/{vl_dice:.4f}")

        if vl_iou > best_iou:
            best_iou = vl_iou
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, "best_model.pth"))
            print(f"  -> New best val IoU: {best_iou:.4f}  (checkpoint saved)")

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "last_model.pth"))

    # ── Evaluate on test set using the best checkpoint ────────────────────────
    best_ckpt = os.path.join(CKPT_DIR, "best_model.pth")
    model.load_state_dict(
        torch.load(best_ckpt, map_location=DEVICE, weights_only=True))
    te_loss, te_iou, te_dice = run_epoch(
        model, test_loader, criterion, optimizer, DEVICE, training=False)
    print(f"\n=== Test set results (best checkpoint) ===")
    print(f"Loss: {te_loss:.4f}  IoU: {te_iou:.4f}  Dice: {te_dice:.4f}")

    # ── Plot training curves ──────────────────────────────────────────────────
    epochs_range = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (tr_key, vl_key), title in zip(
        axes,
        [("train_loss", "val_loss"),
         ("train_iou",  "val_iou"),
         ("train_dice", "val_dice")],
        ["Loss", "IoU", "Dice Score"]
    ):
        ax.plot(epochs_range, history[tr_key], label="Train")
        ax.plot(epochs_range, history[vl_key], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Training curves saved to {plot_path}")


if __name__ == "__main__":
    main()
