import torch
import numpy as np
import config
import env_utils
import paths
import utils
from collections import deque
from functools import partial
import torch.nn.functional as F
from torch.distributions import Categorical
from omegaconf import OmegaConf
import logging
import datetime
import os
from models import LinearDecoder
from data_loader import normalize_obs

# Создаем директорию для логов, если её нет
os.makedirs("logs", exist_ok=True)

def setup_logger(exp_name=None):
    """Настройка логгера с правильным именем файла"""
    # Удаляем все существующие обработчики
    logger = logging.getLogger()
    logger.handlers = []
    
    # Создаем новую конфигурацию
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    
    # Файловый обработчик
    if exp_name:
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"logs/evaluate_{exp_name}_{now_str}.log"
    else:
        log_filename = "logs/evaluate.log"
    
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, log_filename

# Инициализируем базовый логгер
logger, _ = setup_logger()

def evaluate_agent(policy, decoder, env_name, num_episodes=100, device='cuda'):
    """
    Evaluate agent in environment for given number of episodes.
    
    Args:
        policy: trained policy network (generates latent actions)
        decoder: decoder network (converts latent actions to real actions)
        env_name: name of the procgen environment
        num_episodes: number of episodes to evaluate
        device: device to run policy on
    
    Returns:
        mean_return: average return across all episodes
        std_return: standard deviation of returns
        returns: list of all episode returns
    """
    env = env_utils.setup_procgen_env(
        num_envs=1,
        env_id=env_name,
        gamma=0.99
    )
    
    policy.eval()
    decoder.eval()
    returns = []
    
    logger.info(f"Starting evaluation for {num_episodes} episodes in {env_name}")
    
    for episode in range(num_episodes):
        obs = env.reset()
        obs = torch.from_numpy(obs).permute((0, 3, 1, 2)).to(device)
        obs = normalize_obs(obs)
        episode_return = 0
        done = [False]
        step_count = 0
        
        while not done[0]:
            with torch.no_grad():
                # Get latent actions from policy
                latent_actions = policy(obs)
                
                # Decode latent actions to real action logits
                action_logits = decoder(latent_actions)
                
                # Sample action
                dist = Categorical(logits=action_logits)
                action = dist.probs.argmax(dim=-1)
            
            next_obs, reward, done, info = env.step(action.cpu().numpy())
            episode_return += reward[0]
            next_obs = torch.from_numpy(next_obs).permute((0, 3, 1, 2)).to(device)
            obs = normalize_obs(next_obs)
            step_count += 1
        
        returns.append(episode_return)
    
    env.close()
    
    sum_return = np.sum(returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Логируем результаты оценки
    logger.info("="*50)
    logger.info("EVALUATION RESULTS:")
    logger.info(f"Environment: {env_name}")
    logger.info(f"Number of episodes: {num_episodes}")
    logger.info(f"Total Return: {sum_return:.2f}")
    logger.info(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    logger.info(f"Min Return: {np.min(returns):.2f}")
    logger.info(f"Max Return: {np.max(returns):.2f}")
    logger.info(f"All returns: {returns}")
    logger.info("="*50)
    
    return mean_return, std_return, returns


def load_and_evaluate_policy(exp_name, npz_number, env_name, num_episodes=100, device='cuda', decoder_path=None):
    logger.info(f"Loading latent policy from experiment: {exp_name}")
    
    # First load the latent policy
    latent_policy_path = paths.get_latent_policy_path(exp_name)
    logger.info(f"Loading latent policy from: {latent_policy_path}")
    latent_state_dicts = torch.load(latent_policy_path, map_location='cpu', weights_only=False)
    
    cfg = latent_state_dicts["cfg"]
    
    if isinstance(cfg, dict) and exp_name in cfg:
        cfg = cfg[exp_name]
    elif hasattr(cfg, 'keys') and exp_name in cfg.keys():
        cfg = cfg[exp_name]
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    default_cfg = OmegaConf.load("config.yaml")
    default_cfg_dict = OmegaConf.to_container(default_cfg, resolve=True)
    
    merged_cfg = {**default_cfg_dict, **cfg_dict}
    
    cfg = config.get(base_cfg=OmegaConf.create(merged_cfg), use_cli_args=False)
    
    # Create and load the latent policy
    policy = utils.create_policy(
        cfg.model,
        action_dim=cfg.model.la_dim,
        state_dict=latent_state_dicts["policy"],
        strict_loading=True,
    )
    policy = policy.to(device)
    policy.eval()
    
    # Now load the decoder
    if decoder_path is None:
        decoder_path = paths.get_decoded_policy_path(exp_name, n=npz_number)
    logger.info(f"Loading decoder from: {decoder_path}")
    
    try:
        decoder_checkpoint = torch.load(decoder_path, map_location='cpu', weights_only=False)
        decoder = LinearDecoder().to(device)
        decoder.load_state_dict(decoder_checkpoint['decoder_state_dict'])
        decoder.eval()
        logger.info(f"Successfully loaded decoder trained on dataset size {npz_number}")
    except FileNotFoundError:
        logger.error(f"Decoder checkpoint not found at {decoder_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading decoder: {str(e)}")
        raise
    
    logger.info(f"Starting evaluation on device: {device}")
    mean_return, std_return, returns = evaluate_agent(
        policy, decoder, env_name, num_episodes, device
    )
    logger.info(f"Evaluation complete.")
    
    # Сохраняем результаты в отдельный файл
    results_filename = f"logs/results_{exp_name}_n{npz_number}_{env_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_filename, 'w') as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Decoder dataset size: {npz_number}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Number of episodes: {num_episodes}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}\n")
        f.write(f"Min Return: {np.min(returns):.2f}\n")
        f.write(f"Max Return: {np.max(returns):.2f}\n")
        f.write(f"All returns: {returns}\n")
    logger.info(f"Results saved to: {results_filename}")
    
    return mean_return, std_return, returns


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python evaluate.py <exp_name> <npz_number> <env_name> [num_episodes] [device]")
        print("Example: python evaluate.py my_experiment 256 coinrun 100 cuda:0")
        sys.exit(1)
    
    exp_name = sys.argv[1]
    npz_number = int(sys.argv[2])
    env_name = sys.argv[3]
    num_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    device = sys.argv[5] if len(sys.argv) > 5 else 'cuda'

    # Переинициализируем логгер с правильным именем файла
    logger, log_filename = setup_logger(f"{exp_name}_n{npz_number}")
    
    logger.info("="*50)
    logger.info("STARTING NEW EVALUATION SESSION")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Decoder dataset size: {npz_number}")
    logger.info(f"Environment: {env_name}")
    logger.info(f"Number of episodes: {num_episodes}")
    logger.info(f"Device: {device}")
    logger.info(f"Log file: {log_filename}")
    logger.info("="*50)

    try:
        mean_return, std_return, returns = load_and_evaluate_policy(
            exp_name, npz_number, env_name, num_episodes=10, device=str(device),
            decoder_path=checkpoint_path  # pass this as an extra argument, see below
        )
        logger.info("Evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise