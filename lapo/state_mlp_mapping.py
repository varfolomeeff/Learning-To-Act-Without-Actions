import torch
import numpy as np
import argparse
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import env_utils
from data_loader import normalize_obs

# class StatePredictionModel(nn.Module):
#     def __init__(self, obs_dim=(3, 64, 64), num_actions=15, hidden_dim=512):
#         super().__init__()
#         self.num_actions = num_actions
        
#         # Энкодер (как в исходном коде)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(obs_dim[0], 32, 3, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 3, stride=2),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1)
#         )
        
#         # Голова для действий (как в исходном коде)
#         self.action_head = nn.Sequential(
#             nn.Linear(128, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_actions)
#         )
        
#         # Дополнительный декодер для предсказания next_state
#         self.state_decoder = nn.Sequential(
#             nn.Linear(128 + num_actions, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 128 * 4 * 4),
#             nn.Unflatten(1, (128, 4, 4)),
#             nn.ConvTranspose2d(128, 64, 4, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, 4, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, obs_dim[0], 4, stride=2),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         """Основной forward для предсказания действий (как в исходном коде)"""
#         x = self.encoder(x)
#         x = x.view(x.size(0), -1)
#         return self.action_head(x)
    
#     def predict_next_state(self, x, action):
#         """Дополнительный метод для предсказания следующего состояния"""
#         x = self.encoder(x)
#         x = x.view(x.size(0), -1)
#         action_onehot = F.one_hot(action, num_classes=self.num_actions).float()
#         x = torch.cat([x, action_onehot], dim=1)
#         return self.state_decoder(x)

class StatePredictionModel(nn.Module):
    def __init__(self, obs_dim=(3, 64, 64), num_actions=15, hidden_dim=512, latent_dim=128):
        super().__init__()
        self.num_actions = num_actions
        self.latent_dim = latent_dim

        # Энкодер для изображений (обучается на предсказании латентных действий)
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
        
        # Декодер для предсказания реальных действий по латентным
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def encode(self, x):
        """Кодирование изображения в латентное пространство"""
        return self.encoder(x)
    
    def decode(self, z):
        """Декодирование латентного представления в логиты действий"""
        return self.decoder(z)
    
    def forward(self, x):
        """Полный forward pass"""
        z = self.encode(x)
        return self.decode(z)

# Все остальные функции остаются без изменений
def preprocess_observation(obs):
    """Нормализация и преобразование изображения"""
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)
    if obs.ndim == 4:  # (batch, h, w, c)
        obs = obs.permute(0, 3, 1, 2)  # -> (batch, c, h, w)
    return normalize_obs(obs)

def setup_env(env_name):
    """Создание среды с параметрами по умолчанию"""
    return env_utils.setup_procgen_env(
        num_envs=1,
        env_id=env_name,
        gamma=0.99
    )

def evaluate_agent(policy, env_name, num_episodes=100, device='cuda'):
    """Универсальная функция оценки"""
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

def train_and_evaluate(args):
    """Полный цикл обучения и оценки"""
    # 1. Загрузка данных
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

    # 2. Инициализация модели
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = StatePredictionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. Цикл обучения
    for epoch in range(args.epochs):
        try:
            model.train()
            epoch_loss = 0
            
            for obs_batch, act_batch in loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                
                # Основное обучение предсказанию действий
                logits = model(obs_batch)
                loss = F.cross_entropy(logits, act_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(loader):.4f}")

            # 4. Периодическая оценка и сохранение
            if (epoch + 1) % 200 == 0:
                checkpoint_path = f"checkpoints/{args.exp_name}_epoch{epoch+1}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'cfg': {
                        'obs_dim': (3, 64, 64),
                        'num_actions': 15,
                        'hidden_dim': 512,
                        'env_name': args.env_name
                    }
                }, checkpoint_path)
                
                print("Evaluating current model...")
                evaluate_agent(model, args.env_name, 20, device)

        except Exception as e:
            print(f"Error during epoch {epoch+1}: {str(e)}")
            continue



def train_and_evaluate(args):
    """Полный цикл обучения и оценки"""
    # 1. Загрузка данных
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

    # 2. Инициализация модели
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = StatePredictionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. Цикл обучения
    for epoch in range(args.epochs):
        try:
            model.train()
            epoch_loss = 0
            
            for obs_batch, act_batch in loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                
                # 1. Кодируем наблюдения в латентное пространство
                latent_actions = model.encode(obs_batch)
                
                # 2. Декодируем латентные представления в логиты действий
                action_logits = model.decode(latent_actions)
                
                # 3. Вычисляем потери
                loss = F.cross_entropy(action_logits, act_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(loader):.4f}")

            # 4. Периодическая оценка и сохранение
            if (epoch + 1) % 200 == 0:
                checkpoint_path = f"checkpoints/{args.exp_name}_epoch{epoch+1}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
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
            print(f"Error during epoch {epoch+1}: {str(e)}")
            continue
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--env_name', type=str, default='bigfish')
    parser.add_argument('--npz', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Создание директории для чекпоинтов
    Path("checkpoints").mkdir(exist_ok=True)
    
    train_and_evaluate(args)