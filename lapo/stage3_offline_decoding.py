import config
import doy
import env_utils
import paths
import wandb
import ppo
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import data_loader
import random
import numpy as np
from data_loader import normalize_obs
from doy import PiecewiseLinearSchedule as PLS
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from utils import create_decoder
import os

DEVICE = "cuda:0"

# 16, 64, 256, 1024, 4096
TOTAL_SAMPLES = 4096
SEED = 42

@torch.no_grad()
def evaluate_agent(
    env_name,
    policy, 
    decoder, 
    num_episodes=25, 
):
    env = env_utils.setup_procgen_env(
        num_envs=1,
        env_id=env_name,
        gamma=0.99
    )
    returns = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        obs = torch.from_numpy(obs).permute((0, 3, 1, 2)).to(DEVICE)
        obs = normalize_obs(obs)
        episode_return = 0
        done = [False]
        
        while not done[0]:
            latent_actions = policy(obs)
            action = decoder(latent_actions).argmax(dim=-1)

            next_obs, reward, done, info = env.step(action.cpu().numpy())
            episode_return += reward[0]
            next_obs = torch.from_numpy(next_obs).permute((0, 3, 1, 2)).to(DEVICE)
            obs = normalize_obs(next_obs)
        
        returns.append(episode_return)
    
    env.close()
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    return mean_return, std_return


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Set wandb to use existing account without prompts
    
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_SILENT"] = "true"
    
    wandb.init(
        entity="sergeydavidenko3461",
        project="airi-school-lapo",
        group="offline-decoder",
        name=f"total-samples-{TOTAL_SAMPLES}-{SEED}"
    )

    state_dict = torch.load(
        "/home/user6/projects/lapo/lapo/exp_results/0_maze_second_run/latent_policy.pt",
        weights_only=False,
    )
    cfg = config.get(base_cfg=state_dict["cfg"], reload_keys=["stage3", "mlp_mapping"])
    
    policy = utils.create_policy(
        cfg.model,
        action_dim=cfg.model.la_dim,
        state_dict=state_dict["policy"],
        strict_loading=True,
    ).to(DEVICE)
    policy.eval()

    decoder = utils.create_decoder(
        in_dim=cfg.model.la_dim,
        out_dim=cfg.model.ta_dim,
    ).to(DEVICE)
    optim = torch.optim.Adam(decoder.parameters(), lr=3e-4)

    # train_data = np.load("expert_data/bigfish/test/5.npz")
    train_data = np.load(f"offline_decoder_data/bigfish_{TOTAL_SAMPLES}.npz")

    obs_data = torch.tensor(train_data["obs"]).permute(0, 3, 1, 2)
    obs_data = normalize_obs(obs_data)
    ta_data = torch.tensor(train_data["ta"]).long()

    dataset = torch.utils.data.TensorDataset(
        obs_data, 
        ta_data
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True,
    )

    total_update_steps = 700
    epochs = total_update_steps // len(dataloader)

    total_steps = 0
    for epoch in range(epochs):
        for batch in dataloader:
            total_steps += 1
            obs, true_action = [b.to(DEVICE) for b in batch]

            with torch.no_grad():
                pred_latent_action = policy(obs)
            
            pred_true_action = decoder(pred_latent_action)
            loss = F.cross_entropy(pred_true_action, true_action, label_smoothing=0.05)
            acc = (pred_true_action.argmax(-1) == true_action).float().mean()
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            wandb.log({
                "loss": loss.item(),
                "acc": acc.item(),
                "epoch": epoch,
            })
            
            if total_steps % 100 == 0:
                mean_returns, std_returns = evaluate_agent(
                    env_name=cfg.env_name,
                    policy=policy,
                    decoder=decoder,
                    num_episodes=25,
                )
                wandb.log({
                    "eval_returns_mean": mean_returns,
                    "eval_returns_std": std_returns,
                    "epoch": epoch,
                })


if __name__ == "__main__":
    main()