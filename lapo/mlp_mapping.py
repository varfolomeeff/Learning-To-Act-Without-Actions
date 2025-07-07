import sys 
import doy

import numpy as np, glob, os, random, tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import config, paths
import utils
from models import LinearDecoder


sys.argv = [arg for arg in sys.argv if not arg.startswith("gpu=")]

# get gpu from command line
def get_arg(name, default):
    for arg in sys.argv:
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return default

gpu = int(get_arg("gpu", 0))
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

state_dicts = torch.load(
    paths.get_latent_policy_path(config.get().exp_name),
    map_location='cpu',
    weights_only=False
)

print(state_dicts.keys())

cfg = config.get(base_cfg=state_dicts["cfg"], reload_keys=["mlp_mapping"])
if cfg.stage_exp_name is None:
    cfg.stage_exp_name = ""
cfg.stage_exp_name += doy.random_proquint(1)
# doy.print("[bold green]Running LAPO mlp mapping (latent policy decoding) with config:")
# config.print_cfg(cfg)

policy = utils.create_policy(cfg.model,
                             action_dim=cfg.model.la_dim,
                             state_dict=state_dicts["policy"],
                             strict_loading=True)
for p in policy.parameters():           
    p.requires_grad_(False)

print(dir(policy))

policy.decoder = LinearDecoder().to(device)

data_npz = np.load(cfg.mlp_mapping.offline_data)
dataset  = torch.utils.data.TensorDataset(
              torch.tensor(data_npz["obs"]).float()/255.,
              torch.tensor(data_npz["ta"]).long())
loader   = torch.utils.data.DataLoader(dataset,
              batch_size=256, shuffle=True, drop_last=True)

opt = torch.optim.Adam(policy.decoder.parameters(), lr=1e-3)
for epoch in range(cfg.mlp_mapping.epochs):         
    for obs, ta in loader:
        obs = obs.to(device)
        with torch.no_grad():
            z = policy.encoder(obs)           
        logits = policy.decoder(z)
        loss = torch.nn.functional.cross_entropy(logits, ta.to(device))
        opt.zero_grad(); loss.backward(); opt.step()

checkpoint_path = paths.get_decoded_policy_path(cfg.exp_name)
torch.save(policy.decoder.state_dict(), checkpoint_path)
print(f"[bold green]Saved decoder checkpoint to {checkpoint_path}")


