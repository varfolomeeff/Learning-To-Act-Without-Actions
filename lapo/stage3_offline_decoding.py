import torch
import numpy as np
import argparse
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import env_utils
from data_loader import normalize_obs
import wandb
from torch.utils.data import DataLoader, TensorDataset
import random

class StatePredictionModel(nn.Module):
    def __init__(self, obs_dim=(3, 64, 64), num_actions=15, hidden_dim=512, latent_dim=128):
        super().__init__()
        self.num_actions = num_actions
        self.latent_dim = latent_dim

        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_dim[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim)
        )
        
        # Декодер действий
        self.action_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Декодер состояний (опционально)
        self.state_decoder = None

    def encode(self, x):
        return self.encoder(x)
    
    def decode_action(self, z):
        return self.action_decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode_action(z)

def preprocess_observation(obs):
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)
    if obs.ndim == 4:
        obs = obs.permute(0, 3, 1, 2)
    return normalize_obs(obs)

def setup_env(env_name):
    return env_utils.setup_procgen_env(
        num_envs=1,
        env_id=env_name,
        gamma=0.99
    )

@torch.no_grad()
def evaluate_agent(policy, env_name, num_episodes=25, device='cuda'):
    """Обновленная функция оценки с метриками из второго скрипта"""
    try:
        env = setup_env(env_name)
        policy.eval()
        returns = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            obs = torch.from_numpy(obs).permute((0, 3, 1, 2)).to(device)
            obs = normalize_obs(obs)
            episode_return = 0
            done = [False]
            
            while not done[0]:
                logits = policy(obs)
                action = logits.argmax(dim=-1)
                
                next_obs, reward, done, _ = env.step(action.cpu().numpy())
                episode_return += reward[0]
                next_obs = torch.from_numpy(next_obs).permute((0, 3, 1, 2)).to(device)
                obs = normalize_obs(next_obs)
            
            returns.append(episode_return)
        
        env.close()
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        return mean_return, std_return, returns

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        if 'env' in locals():
            env.close()
        return 0, 0, []

def train_encoder(args):
    """Этап 1: Обучение энкодера с логированием в WandB"""
    print("=== Этап 1: Предварительное обучение энкодера ===")
    
    try:
        npz_path = f"offline_decoder_data/{args.env_name}_{args.npz}.npz"
        data = np.load(npz_path)
        obs = torch.tensor(data["obs"])
        actions = torch.tensor(data["ta"]).long()
        
        dataset = TensorDataset(
            preprocess_observation(obs),
            actions
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        return None

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = StatePredictionModel().to(device)
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=args.lr)

    # Инициализация WandB
    wandb.init(
        project=args.wandb_project,
        group=args.exp_name,
        name=f"{args.exp_name}_encoder",
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "encoder_epochs": args.encoder_epochs,
            "env_name": args.env_name
        }
    )

    for epoch in range(args.encoder_epochs):
        try:
            model.train()
            epoch_loss = 0
            epoch_acc = 0
            
            for obs_batch, act_batch in loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                
                # Обучаем энкодер через задачу предсказания действий
                z = model.encode(obs_batch)
                action_logits = model.decode_action(z)
                loss = F.cross_entropy(action_logits, act_batch, label_smoothing=0.05)
                
                # Вычисляем accuracy
                pred_actions = action_logits.argmax(dim=-1)
                acc = (pred_actions == act_batch).float().mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
            avg_loss = epoch_loss / len(loader)
            avg_acc = epoch_acc / len(loader)
            
            print(f"Encoder Epoch {epoch+1}/{args.encoder_epochs}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
            
            # Логирование в WandB
            wandb.log({
                "encoder_loss": avg_loss,
                "encoder_acc": avg_acc,
                "epoch": epoch
            })

            if (epoch + 1) % args.eval_interval == 0:
                checkpoint_path = f"checkpoints/encoder/{args.exp_name}_encoder_epoch{epoch+1}.pt"
                torch.save({
                    'encoder_state_dict': model.encoder.state_dict(),
                    'cfg': {
                        'obs_dim': (3, 64, 64),
                        'latent_dim': 128,
                        'env_name': args.env_name
                    }
                }, checkpoint_path)

        except Exception as e:
            print(f"Error during encoder epoch {epoch+1}: {str(e)}")
            continue

    wandb.finish()
    return model.encoder

def train_decoder(args, encoder):
    """Этап 2: Обучение декодера с логированием в WandB"""
    if encoder is None:
        print("Encoder is None, skipping decoder training")
        return
    
    print("\n=== Этап 2: Обучение декодера действий ===")
    
    try:
        npz_path = f"offline_decoder_data/{args.env_name}_{args.npz}.npz"
        data = np.load(npz_path)
        obs = torch.tensor(data["obs"])
        actions = torch.tensor(data["ta"]).long()
        
        dataset = TensorDataset(
            preprocess_observation(obs),
            actions
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = StatePredictionModel().to(device)
    model.encoder.load_state_dict(encoder.state_dict())
    
    # Замораживаем энкодер
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.action_decoder.parameters(), lr=args.lr)

    # Инициализация WandB
    wandb.init(
        project=args.wandb_project,
        group=args.exp_name,
        name=f"{args.exp_name}_decoder",
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "decoder_epochs": args.decoder_epochs,
            "env_name": args.env_name
        }
    )

    for epoch in range(args.decoder_epochs):
        try:
            model.train()
            epoch_loss = 0
            epoch_acc = 0
            
            for obs_batch, act_batch in loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                
                # 1. Кодируем наблюдения (энкодер заморожен)
                with torch.no_grad():
                    z = model.encode(obs_batch)
                
                # 2. Декодируем в действия
                action_logits = model.decode_action(z)
                
                # 3. Вычисляем потерю и accuracy
                loss = F.cross_entropy(action_logits, act_batch, label_smoothing=0.05)
                pred_actions = action_logits.argmax(dim=-1)
                acc = (pred_actions == act_batch).float().mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
            avg_loss = epoch_loss / len(loader)
            avg_acc = epoch_acc / len(loader)
            
            print(f"Decoder Epoch {epoch+1}/{args.decoder_epochs}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
            
            # Логирование в WandB
            wandb.log({
                "decoder_loss": avg_loss,
                "decoder_acc": avg_acc,
                "epoch": epoch
            })

            if (epoch + 1) % args.eval_interval == 0:
                checkpoint_path = f"checkpoints/decoder/{args.exp_name}_decoder_epoch{epoch+1}.pt"
                torch.save({
                    'decoder_state_dict': model.action_decoder.state_dict(),
                    'encoder_state_dict': model.encoder.state_dict(),
                    'cfg': {
                        'obs_dim': (3, 64, 64),
                        'num_actions': 15,
                        'hidden_dim': 512,
                        'latent_dim': 128,
                        'env_name': args.env_name
                    }
                }, checkpoint_path)
                
                # Оценка модели
                mean_return, std_return, returns = evaluate_agent(
                    model, args.env_name, args.eval_episodes, device
                )
                
                wandb.log({
                    "eval_mean_return": mean_return,
                    "eval_std_return": std_return,
                    "epoch": epoch
                })

        except Exception as e:
            print(f"Error during decoder epoch {epoch+1}: {str(e)}")
            continue

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--env_name', type=str, default='bigfish')
    parser.add_argument('--npz', type=int, default=256)
    parser.add_argument('--encoder_epochs', type=int, default=1000)
    parser.add_argument('--decoder_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--eval_episodes', type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_project', type=str, default='rl-project')
    
    args = parser.parse_args()
    
    # Установка случайных seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    Path("checkpoints/encoder").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/decoder").mkdir(parents=True, exist_ok=True)
    
    # Этап 1: Обучение энкодера
    trained_encoder = train_encoder(args)
    
    # Этап 2: Обучение декодера
    if trained_encoder is not None:
        train_decoder(args, trained_encoder)
    else:
        print("Skipping decoder training due to encoder training failure")