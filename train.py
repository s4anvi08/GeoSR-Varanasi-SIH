"""
Train GeoSR on your own images.

Usage:
    python train.py --data_dir ./data/hr_images --epochs 30 --scale 4

Since we (almost certainly) don't have paired true 10m/<4m Sentinel-2
imagery for training, we synthesize training pairs the standard way
super-resolution papers do it: take your high-resolution images as the
"ground truth" (HR), downsample them by `scale` to create the
"medium-resolution" input (LR), and train the network to reverse that
downsampling. This is a legitimate, standard SR training recipe (used by
SRCNN/EDSR/ESRGAN papers) — swap in real paired Sentinel-2 / commercial
high-res tiles for production training.
"""
import argparse
import glob
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from model import GeoSR


class PatchSRDataset(Dataset):
    """Randomly crops HR patches from your images, downsamples for LR input."""

    def __init__(self, image_paths, scale=4, hr_patch=96, patches_per_image=40):
        self.paths = image_paths
        self.scale = scale
        self.hr_patch = hr_patch
        self.patches_per_image = patches_per_image
        self.images = []
        for p in self.paths:
            img = Image.open(p).convert("RGB")
            # upscale tiny images so we can always crop a patch
            if min(img.size) < hr_patch:
                ratio = (hr_patch + 4) / min(img.size)
                img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.BICUBIC)
            self.images.append(img)

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx):
        img = self.images[idx % len(self.images)]
        w, h = img.size
        x = random.randint(0, w - self.hr_patch)
        y = random.randint(0, h - self.hr_patch)
        hr = img.crop((x, y, x + self.hr_patch, y + self.hr_patch))

        # random flip/rotate augmentation
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)

        lr_size = self.hr_patch // self.scale
        lr = hr.resize((lr_size, lr_size), Image.BICUBIC)

        hr_arr = np.asarray(hr, dtype=np.float32) / 255.0
        lr_arr = np.asarray(lr, dtype=np.float32) / 255.0
        hr_t = torch.from_numpy(hr_arr).permute(2, 0, 1)
        lr_t = torch.from_numpy(lr_arr).permute(2, 0, 1)
        return lr_t, hr_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data/hr_images")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--hr_patch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="./checkpoints/geosr.pt")
    args = ap.parse_args()

    paths = sorted(sum([glob.glob(os.path.join(args.data_dir, ext))
                         for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")], []))
    if not paths:
        raise SystemExit(f"No images found in {args.data_dir}. Drop your Sentinel-2 / "
                          f"satellite JPEG/PNG/TIFF tiles there first.")
    print(f"Found {len(paths)} source image(s): {[os.path.basename(p) for p in paths]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    dataset = PatchSRDataset(paths, scale=args.scale, hr_patch=args.hr_patch)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = GeoSR(scale=args.scale).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()  # L1 is standard for SR, sharper results than MSE

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for lr_t, hr_t in loader:
            lr_t, hr_t = lr_t.to(device), hr_t.to(device)
            opt.zero_grad()
            sr_t = model(lr_t)
            loss = criterion(sr_t, hr_t)
            loss.backward()
            opt.step()
            running += loss.item()
        avg = running / len(loader)
        print(f"epoch {epoch:3d}/{args.epochs}  L1 loss {avg:.4f}")

    torch.save({"state_dict": model.state_dict(), "scale": args.scale}, args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
