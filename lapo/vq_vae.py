import argparse
import pathlib
from collections import deque
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms.functional import resize


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0  # (3,64,64)
    return t


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
        )

    def forward(self, x):
        return x + self.net(x)

class Encoder(nn.Module):
    def __init__(self, z_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 64→32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), # 32→16
            ResidualBlock(64), ResidualBlock(64),
            nn.Conv2d(64, z_channels, 1),          # 16→16 latent map
        )
    def forward(self, x):
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, z_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_channels*2, 64, 4, 2, 1), # 16→32
            nn.ReLU(inplace=True),
            ResidualBlock(64), ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 32→64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),
        )
    def forward(self, z_t, z_tm1):
        return self.net(torch.cat([z_t, z_tm1], dim=1))

class VectorQuantizerEMA(nn.Module):
    def __init__(self, K: int, z_channels: int, beta=0.25, decay=0.99):
        super().__init__()
        self.K = K
        self.beta = beta
        self.decay = decay
        self.register_buffer('embed', torch.randn(K, z_channels))
        self.register_buffer('cluster_size', torch.zeros(K))

    @torch.no_grad()
    def _update(self, enc_flat, codes):
        # EMA update
        onehot = F.one_hot(codes, self.K).type_as(enc_flat)
        self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1-self.decay)
        embed_sum = enc_flat.t() @ onehot
        self.embed.mul_(self.decay).add_(embed_sum.t(), alpha=1-self.decay)
        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + 1e-5) / (n + self.K * 1e-5) * n
        self.embed.copy_(self.embed / cluster_size.unsqueeze(1))

    def forward(self, z: torch.Tensor):
        B, C, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, C)  # (BHW, C)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2*flat @ self.embed.t()
                + self.embed.pow(2).sum(1))          # (BHW,K)
        codes = dist.argmin(1)                       # (BHW,)
        z_q = self.embed[codes].view(B, H, W, C).permute(0, 3, 1, 2)

        if self.training:
            self._update(flat.detach(), codes.detach())

        loss = self.beta * (z_q.detach() - z).pow(2).mean() + (z_q - z.detach()).pow(2).mean()
        z_q = z + (z_q - z).detach()  # straight‑through
        return z_q, codes.view(B, H, W), loss

class VQVAE(nn.Module):
    def __init__(self, z_channels=64, K=64):
        super().__init__()
        self.enc = Encoder(z_channels)
        self.vq  = VectorQuantizerEMA(K, z_channels)
        self.dec = Decoder(z_channels)

    def forward(self, o_t, o_tm1):
        z_t   = self.enc(o_t)
        z_tm1 = self.enc(o_tm1)
        zq_t,  codes, loss_vq  = self.vq(z_t)
        zq_tm1 = self.vq.embed[codes.view(-1)].view_as(zq_t).detach()  # no grad
        recon = self.dec(zq_t, zq_tm1)
        loss_rec = F.mse_loss(recon, o_t)
        return recon, codes, loss_rec + loss_vq


class ProcgenNPZ():
    def __init__(self, root_dir, want_code=False):
        self.want_code = want_code
        self.files = sorted(glob.glob(f"{root_dir}/**/*.npz", recursive=True))
        self.index = []
        for fid, f in enumerate(self.files):
            n = np.load(f, mmap_mode="r")["obs"].shape[0]
            self.index.extend([(fid, i) for i in range(n)])
        print(f"[dataset] {len(self.files)} files ⇒ {len(self.index)} samples")

    def __len__(self):  return len(self.index)

    def __getitem__(self, idx):
        fid, i = self.index[idx]
        d = np.load(self.files[fid], mmap_mode="r")
        obs = d["obs"][i]
        if self.want_code:
            code = d["code"][i]
            act  = d["ta"][i]
            return obs, code, act
        else:
            obs_tp1 = d["obs"][i+1] if i+1 < d["obs"].shape[0] else d["obs"][i]
            return obs, obs_tp1


def train_vqvae(npz_path: str, out_ckpt: str, epochs=10, bs=128):
    ds  = ProcgenNPZ(npz_path)
    dl  = DataLoader(ds, bs, shuffle=True, num_workers=4, pin_memory=True)
    vqvae = VQVAE(z_channels=64, K=64).to(DEVICE)
    opt   = torch.optim.Adam(vqvae.parameters(), lr=3e-4)

    for ep in range(epochs):
        vqvae.train()
        running = 0.0
        for o, o1 in dl:
            o, o1 = o.to(DEVICE), o1.to(DEVICE)
            recon, codes, loss = vqvae(o, o1)
            print(f"Loss values: recon - {recon}, vq - {loss}")
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * o.size(0)
        print(f"VQ‑VAE epoch {ep+1}/{epochs}  loss={running/len(ds):.4f}")

    torch.save({'model': vqvae.state_dict()}, out_ckpt)
    print("saved", out_ckpt)
    return vqvae


class BCData(torch.utils.data.Dataset):
    def __init__(self, npz_path: str, vqvae: VQVAE):
        d = np.load(npz_path)
        self.obs = d["obs"]          # (N,64,64,3)
        self.ta  = d["ta"]
        self.vq  = vqvae.eval().to(DEVICE)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        o = preprocess(self.obs[i]).unsqueeze(0).to(DEVICE)  # (1,3,64,64)
        with torch.no_grad():
            z = self.vq.enc(o)                               # (1,C,16,16)
            _, code_map, _ = self.vq.vq(z)                  # (1,16,16)
        code_id = code_map.view(-1)[0]
        return torch.tensor(code_id, dtype=torch.long), torch.tensor(int(self.ta[i]), dtype=torch.long)

class Code2Action(nn.Module):
    def __init__(self, K, n_act=15):
        super().__init__()
        self.embed = nn.Embedding(K, n_act)
    def forward(self, code_idx):
        return self.embed(code_idx)
    def __init__(self, K):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512), nn.ReLU(),
        )
        self.head = nn.Linear(512, K)
    def forward(self, x):
        return self.head(self.cnn(x))


def train_bc(npz_path: str, vq_ckpt: str, out_ckpt: str, epochs=3, bs=1024):
    vqvae = VQVAE(z_channels=64, K=64).to(DEVICE)
    vqvae.load_state_dict(torch.load(vq_ckpt, map_location=DEVICE)['model'])
    bc_ds = BCData(npz_path, vqvae)
    dl    = DataLoader(bc_ds, bs, shuffle=True, num_workers=4, pin_memory=True)

    decoder = Code2Action(K=64).to(DEVICE)
    opt     = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    crit    = nn.CrossEntropyLoss()

    for ep in range(epochs):
        decoder.train(); running=0.0
        for code_idx, act in dl:
            code_idx, act = code_idx.to(DEVICE), act.to(DEVICE)
            logits = decoder(code_idx)
            loss = crit(logits, act)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * code_idx.size(0)
        print(f"BC‑decoder epoch {ep+1}/{epochs}  loss={running/len(bc_ds):.4f}")
    torch.save({'decoder': decoder.state_dict()}, out_ckpt)
    print("saved", out_ckpt)
    print("saved", out_ckpt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="expert demo npz")
    ap.add_argument("--stage", choices=["vqvae","bc"], required=True)
    ap.add_argument("--vq_ckpt", default="vqvae.pt")
    ap.add_argument("--bc_ckpt", default="bc_policy.pt")
    args = ap.parse_args()

    pathlib.Path("outputs").mkdir(exist_ok=True)

    if args.stage == "vqvae":
        train_vqvae(args.npz, args.vq_ckpt)
    else:
        train_bc(args.npz, args.vq_ckpt, args.bc_ckpt)
