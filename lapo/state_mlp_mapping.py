import torch
import numpy as np
import argparse
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import env_utils
from data_loader import normalize_obs

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

def evaluate_agent(policy, env_name, num_episodes=100, device='cuda'):
    try:
        env = setup_env(env_name)
        policy.eval()
        returns = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            obs = preprocess_observation(obs).to(device)
            episode_return = 0
            done = [False]
            
            while not done[0]:
                with torch.no_grad():
                    logits = policy(obs)
                    probs = Categorical(logits=logits)
                    action = probs.sample()
                
                next_obs, reward, done, _ = env.step(action.cpu().numpy())
                obs = preprocess_observation(next_obs).to(device)
                episode_return += reward[0]
            
            returns.append(episode_return)
            print(f"Episode {episode + 1}/{num_episodes}: Return = {episode_return:.2f}")
        
        env.close()
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        print(f"\nEvaluation Results:")
        print(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")
        return mean_return, std_return, returns
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        if 'env' in locals():
            env.close()
        return 0, 0, []

def train_encoder(args):
    """Этап 1: Обучение энкодера (упрощенный вариант, так как нет next_obs)"""
    print("=== Этап 1: Предварительное обучение энкодера ===")
    
    try:
        npz_path = f"offline_decoder_data/{args.env_name}_{args.npz}.npz"
        data = np.load(npz_path)
        obs = torch.tensor(data["obs"])
        actions = torch.tensor(data["ta"]).long()
        
        dataset = torch.utils.data.TensorDataset(
            preprocess_observation(obs),
            actions
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        return None

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = StatePredictionModel().to(device)
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=1e-3)

    # Упрощенное обучение энкодера (так как нет next_obs)
    for epoch in range(args.encoder_epochs):
        try:
            model.train()
            epoch_loss = 0
            
            for obs_batch, act_batch in loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                
                # Обучаем энкодер через задачу предсказания действий
                z = model.encode(obs_batch)
                action_logits = model.decode_action(z)
                loss = F.cross_entropy(action_logits, act_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Encoder Epoch {epoch+1}/{args.encoder_epochs}, Loss: {epoch_loss/len(loader):.4f}")

            if (epoch + 1) % 200 == 0:
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

    return model.encoder

def train_decoder(args, encoder):
    """Этап 2: Обучение декодера действий с замороженным энкодером"""
    if encoder is None:
        print("Encoder is None, skipping decoder training")
        return
    
    print("\n=== Этап 2: Обучение декодера действий ===")
    
    try:
        npz_path = f"offline_decoder_data/{args.env_name}_{args.npz}.npz"
        data = np.load(npz_path)
        obs = torch.tensor(data["obs"])
        actions = torch.tensor(data["ta"]).long()
        
        dataset = torch.utils.data.TensorDataset(
            preprocess_observation(obs),
            actions
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = StatePredictionModel().to(device)
    model.encoder.load_state_dict(encoder.state_dict())
    
    # Замораживаем энкодер
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.action_decoder.parameters(), lr=1e-3)

    for epoch in range(args.decoder_epochs):
        try:
            model.train()
            epoch_loss = 0
            
            for obs_batch, act_batch in loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                
                # 1. Кодируем наблюдения (энкодер заморожен)
                with torch.no_grad():
                    z = model.encode(obs_batch)
                
                # 2. Декодируем в действия
                action_logits = model.decode_action(z)
                
                # 3. Вычисляем потерю
                loss = F.cross_entropy(action_logits, act_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Decoder Epoch {epoch+1}/{args.decoder_epochs}, Loss: {epoch_loss/len(loader):.4f}")

            if (epoch + 1) % 200 == 0:
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
                
                print("Evaluating current model...")
                evaluate_agent(model, args.env_name, 20, device)

        except Exception as e:
            print(f"Error during decoder epoch {epoch+1}: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--env_name', type=str, default='bigfish')
    parser.add_argument('--npz', type=int, default=256)
    parser.add_argument('--encoder_epochs', type=int, default=1000)
    parser.add_argument('--decoder_epochs', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    Path("checkpoints/encoder").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/decoder").mkdir(parents=True, exist_ok=True)
    
    # Этап 1: Обучение энкодера
    trained_encoder = train_encoder(args)
    
    # Этап 2: Обучение декодера
    if trained_encoder is not None:
        train_decoder(args, trained_encoder)
    else:
        print("Skipping decoder training due to encoder training failure")