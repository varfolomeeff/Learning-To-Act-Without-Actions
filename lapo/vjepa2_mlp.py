import argparse
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoVideoProcessor
from tqdm.auto import tqdm


class ProcgenNPZ(Dataset):
    def __init__(
        self,
        file: str | Path,
        num_frames: int = 1,
        image_size: int = 256,
        dtype: Literal["float32", "float16"] = "float32",
    ):
        super().__init__()
        with np.load(file, allow_pickle=False) as d:
            self.images = d["obs"].copy()
            self.actions = d["ta"].astype(np.int64, copy=True)
        self.num_frames = num_frames
        self.processor = AutoVideoProcessor.from_pretrained(
            "facebook/vjepa2-vitl-fpc64-256",
            size={"height": image_size, "width": image_size},
            do_center_crop=True,
            do_rescale=True,
            do_normalize=True,
        )
        self.dtype = torch.float16 if dtype == "float16" else torch.float32

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        act = self.actions[idx]
        video = img[np.newaxis, ...] if self.num_frames == 1 else np.repeat(img[np.newaxis, ...], self.num_frames, axis=0)
        px = self.processor(video, return_tensors="pt")["pixel_values_videos"].to(self.dtype).squeeze(0)
        return px, torch.as_tensor(act, dtype=torch.long)


class VJEPAActionPolicy(nn.Module):
    def __init__(self, num_actions: int = 15, train_backbone: bool = False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
        for p in self.backbone.parameters():
            p.requires_grad = train_backbone
        self.embed_dim = self.backbone.config.hidden_size  # 1024 for ViT‑L
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, pixel_values_videos: torch.Tensor):
        with torch.set_grad_enabled(self.backbone.training):
            out = self.backbone(pixel_values_videos=pixel_values_videos, output_hidden_states=False)
        embed = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state.mean(dim=1)
        return self.head(embed)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ProcgenNPZ(
        cfg.data_path,
        num_frames=cfg.num_frames,
        image_size=cfg.image_size,
        dtype="float16" if cfg.mixed_precision else "float32",
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=torch.cuda.is_available() and cfg.workers > 0,
    )

    model = VJEPAActionPolicy(cfg.num_actions, cfg.finetune_backbone).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(dl, desc=f"[epoch {epoch+1}/{cfg.epochs}]", leave=False)
        for px, act in pbar:
            px, act = px.to(device, non_blocking=True), act.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                logits = model(px)
                loss = crit(logits, act)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * px.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"➤ epoch {epoch+1}: mean CE {running/len(ds):.4f}")

        if (epoch + 1) % cfg.save_every == 0 or (epoch + 1) == cfg.epochs:
            ckpt = Path(cfg.output_dir) / f"lapo_vjepa_epoch{epoch+1}.pt"
            torch.save({"model_state_dict": model.state_dict(), "cfg": vars(cfg)}, ckpt)
            print(f"✓ saved {ckpt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_actions", type=int, default=15)
    ap.add_argument("--num_frames", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--finetune_backbone", action="store_true")
    ap.add_argument("--mixed_precision", action="store_true")
    ap.add_argument("--save_every", type=int, default=20)
    cfg = ap.parse_args()

    print("▶ Config:")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")
    train(cfg)
