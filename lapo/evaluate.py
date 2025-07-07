import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Literal
import env_utils
from collections import deque
from functools import partial
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import nn
from transformers import AutoModel, AutoVideoProcessor


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


def preprocess_observation(obs, processor, num_frames=1, image_size=256, dtype=torch.float32):
    if obs.ndim == 4:  # (batch, height, width, channels)
        obs = obs[0]  # Take first (and only) observation
    
    video = obs[np.newaxis, ...] if num_frames == 1 else np.repeat(obs[np.newaxis, ...], num_frames, axis=0)
    
    processed = processor(video, return_tensors="pt")["pixel_values_videos"].to(dtype)
    
    return processed


def evaluate_agent(policy, env_name, num_episodes=100, device='cuda', num_frames=1, image_size=256):
    env = env_utils.setup_procgen_env(
        num_envs=1,
        env_id=env_name,
        gamma=0.99
    )
    
    processor = AutoVideoProcessor.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        size={"height": image_size, "width": image_size},
        do_center_crop=True,
        do_rescale=True,
        do_normalize=True,
    )
    
    policy.eval()
    returns = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_return = 0
        done = [False]
        
        while not done[0]:
            with torch.no_grad():
                processed_obs = preprocess_observation(
                    obs, processor, num_frames, image_size, policy.head[0].weight.dtype
                ).to(device)
                
                logits = policy(processed_obs)
                probs = Categorical(logits=logits)
                action = probs.sample()
            
            next_obs, reward, done, info = env.step(action.cpu().numpy())
            episode_return += reward[0]
            obs = next_obs
        
        returns.append(episode_return)
        print(f"Episode {episode + 1}/{num_episodes}: Return = {episode_return:.2f}")
    
    env.close()
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    print(f"\nEvaluation Results:")
    print(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Min Return: {np.min(returns):.2f}")
    print(f"Max Return: {np.max(returns):.2f}")
    
    return mean_return, std_return, returns


def load_and_evaluate_policy(policy_checkpoint_path, env_name, num_episodes=100, device='cuda'):
    checkpoint = torch.load(policy_checkpoint_path, map_location='cpu')
    
    cfg = checkpoint.get('cfg', {})
    
    policy = VJEPAActionPolicy(
        num_actions=cfg.get('num_actions', 15),
        train_backbone=cfg.get('finetune_backbone', False)
    )
    
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy = policy.to(device)
    
    print(f"Loaded policy from: {policy_checkpoint_path}")
    print(f"Config: {cfg}")
    
    mean_return, std_return, returns = evaluate_agent(
        policy, 
        env_name, 
        num_episodes, 
        device,
        num_frames=cfg.get('num_frames', 1),
        image_size=cfg.get('image_size', 256)
    )
    
    return mean_return, std_return, returns


def main():
    parser = argparse.ArgumentParser(description='Evaluate V-JEPA trained policy')
    parser.add_argument('policy_checkpoint_path', type=str, help='Path to the checkpoint file')
    parser.add_argument('env_name', type=str, help='Name of the procgen environment')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    
    args = parser.parse_args()
    
    if not Path(args.policy_checkpoint_path).exists():
        print(f"Error: Checkpoint file not found: {args.policy_checkpoint_path}")
        return
    
    print(f"Evaluating policy from checkpoint: {args.policy_checkpoint_path}")
    print(f"Environment: {args.env_name}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Device: {args.device}")
    print("-" * 50)
    
    mean_return, std_return, returns = load_and_evaluate_policy(
        args.policy_checkpoint_path, args.env_name, args.num_episodes, args.device
    )
    
    return mean_return, std_return, returns


if __name__ == "__main__":
    main()