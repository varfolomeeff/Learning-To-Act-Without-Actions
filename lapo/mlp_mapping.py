import sys 
import doy
import argparse

import numpy as np, glob, os, random, tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import config, paths
import utils
from models import LinearDecoder
from omegaconf import OmegaConf, DictConfig
from evaluate import load_and_evaluate_policy
import wandb


sys.argv = [arg for arg in sys.argv if not arg.startswith("gpu=")]

def get_arg(name, default):
    for arg in sys.argv:
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return default

gpu = int(get_arg("gpu", 1))
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--npz', type=int, default=256, help='Number in the offline_decoder_data/{env_name}_{npz}.npz filename')
parser.add_argument('--env_name', type=str, default='bigfish', help='Environment name')
parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
parser.add_argument('--base_exp_name', type=str, default='bigfish', help='Experiment name to load policy from')
args, unknown = parser.parse_known_args()

env_name = args.env_name
base_exp_name = args.base_exp_name

wandb.init()  # This must be called before using wandb.run

if args.exp_name is None:
    exp_name = f"sweep_{wandb.run.id}"
else:
    exp_name = args.exp_name

print(f"env_name: {env_name}")

def remove_script_keys(d):
    if isinstance(d, dict):
        keys_to_remove = [k for k in d if str(k).startswith('--')]
        for k in keys_to_remove:
            del d[k]
        for v in d.values():
            remove_script_keys(v)
    elif isinstance(d, list):
        for item in d:
            remove_script_keys(item)
    elif isinstance(d, DictConfig):
        keys_to_remove = [k for k in d.keys() if str(k).startswith('--')]
        for k in d.keys():
            if OmegaConf.is_missing(d, k):
                continue
            remove_script_keys(d[k])
        for k in keys_to_remove:
            del d[k]

def print_all_keys(d, prefix=''):
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{prefix}{k}")
            print_all_keys(v, prefix + '  ')
    elif isinstance(d, list):
        for i, item in enumerate(d):
            print_all_keys(item, prefix + f'[{i}] ')

state_dicts = torch.load(
    paths.get_latent_policy_path(base_exp_name),
    map_location='cpu',
    weights_only=False
)
print('All keys in state_dicts["cfg"] after loading:')
# print_all_keys(state_dicts["cfg"])
remove_script_keys(state_dicts["cfg"])  # Clean loaded config

if isinstance(state_dicts["cfg"], dict):
    base_cfg = state_dicts["cfg"]
else:
    base_cfg = OmegaConf.to_container(state_dicts["cfg"], resolve=True)
base_cfg["env_name"] = env_name
if not isinstance(base_cfg, dict):
    base_cfg = OmegaConf.to_container(base_cfg, resolve=True)
remove_script_keys(base_cfg)  # Clean again before merging

print('All keys in base_cfg before merging:')
print_all_keys(base_cfg)

cfg = config.get(base_cfg=base_cfg, reload_keys=["mlp_mapping"], use_cli_args=False)
if cfg.stage_exp_name is None:
    cfg.stage_exp_name = ""
cfg.stage_exp_name += doy.random_proquint(1)

policy = utils.create_policy(cfg.model,
                             action_dim=cfg.model.la_dim,
                             state_dict=state_dicts["policy"],
                             strict_loading=True)
policy = policy.to(device)
policy.eval()

for p in policy.parameters():           
    p.requires_grad_(False)

decoder = LinearDecoder(cfg.model.la_dim, cfg.model.ta_dim).to(device)

npz_number = args.npz
# npz_path = f"offline_decoder_data/{env_name}_{npz_number}.npz"
npz_path = "/home/user6/projects/lapo/lapo/expert_data/bigfish/train/0.npz"
data_npz = np.load(npz_path)

def normalize_obs(obs: torch.Tensor) -> torch.Tensor:
    assert not torch.is_floating_point(obs)
    return obs.float() / 255 - 0.5

obs_data = torch.tensor(data_npz["obs"])
obs_data = normalize_obs(obs_data)
# Ensure obs_data shape is (batch, height, width, channels) before permute
if obs_data.ndim != 4 or obs_data.shape[-1] not in [1, 3]:
    raise ValueError(f"Unexpected obs_data shape: {obs_data.shape}")
obs_data = obs_data.permute(0, 3, 1, 2)  # (batch, height, width, channels) -> (batch, channels, height, width)

dataset = torch.utils.data.TensorDataset(
    obs_data,
    torch.tensor(data_npz["ta"]).long()
)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=True, 
    drop_last=False
)

print("Dataset size:", len(dataset))

opt = torch.optim.Adam(decoder.parameters(), lr=4e-4)

# Используйте wandb.config для доступа к параметрам sweep
cfg.mlp_mapping.lr = wandb.config.get("learning_rate", 1e-3)
cfg.mlp_mapping.hid_dim = wandb.config.get("hidden_size", 128)
# и т.д. для других параметров

for epoch in range(cfg.mlp_mapping.epochs):         
    epoch_loss = 0
    num_batches = 0

    for obs, ta in loader:
        obs = obs.to(device)
        ta = ta.to(device)

        with torch.no_grad():
            latent_actions = policy(obs)

        action_logits = decoder(latent_actions)

        loss = torch.nn.functional.cross_entropy(action_logits, ta, label_smoothing=0.05)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        num_batches += 1
        
        
        wandb.log({"train_loss": loss.item(), "epoch": epoch})


    avg_loss = epoch_loss / num_batches if num_batches > 0 else float('nan')
    # Calculate test accuracy on the full dataset after the epoch
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for obs, ta in loader:
            obs = obs.to(device)
            ta = ta.to(device)
            latent_actions = policy(obs)
            action_logits = decoder(latent_actions)
            preds = action_logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(ta.cpu())
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        test_acc = (all_preds == all_targets).float().mean().item()
    wandb.log({"test_accuracy": test_acc, "epoch": epoch})
    print(f"Epoch {epoch+1}/{cfg.mlp_mapping.epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    if epoch > 0 and epoch % 200 == 0:
        base_dir = paths.get_experiment_dir(exp_name)
        checkpoint_name = f"decoded_policy_{npz_number}_epoch{epoch}.pt"
        checkpoint_path = base_dir / checkpoint_name
        torch.save({
            'decoder_state_dict': decoder.state_dict(),
            'cfg': cfg
        }, checkpoint_path)
        print(f"Saved decoder checkpoint to {checkpoint_path}")
        # Evaluate after saving checkpoint
        print(f"Evaluating decoder at epoch {epoch}...")
        mean_return, std_return, returns = load_and_evaluate_policy(
            base_exp_name, npz_number, env_name, num_episodes=100, device=str(device),
            decoder_path=checkpoint_path
        )
        print(f"Eval result at epoch {epoch}: mean_return={mean_return:.2f} ± {std_return:.2f}")

        wandb.log({
            
        "mean_return": mean_return,
        "std_return": std_return,
        "min_return": min(returns),
        "max_return": max(returns),
        "all_returns": returns,
        "epoch": epoch
        })