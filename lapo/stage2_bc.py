import config
import data_loader
import doy
import paths
import torch
import torch.nn.functional as F
import utils
from doy import loop
import numpy
from torch.serialization import add_safe_methods
import sys

sys.argv = [rig for arg in sys.argv if not arg.startswith("gpu=")]

add_safe_methods([numpy.core.multiarray.scalar])

def clean_state_dict(state_dict):
    """Removes 'module.' prefix from state_dict keys if present."""
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict

# Загружаем предобученные модели из stage1
state_dicts = torch.load(
    paths.get_models_path(config.get().exp_name),
    map_location='cpu',
    weights_only=False
)
cfg = config.get(base_cfg=state_dicts["cfg"], reload_keys=["stage2", "stage3"])
cfg.stage_exp_name = doy.random_proquint(1)
doy.print("[bold green]Running LAPO stage 2 (REAL action behavior cloning) with config:")
config.print_cfg(cfg)

if state_dicts["step"] != cfg.stage1.steps:
    doy.log(
        f"[bold red]Warning: using IDM/WM from incomplete training run {state_dicts['step']}/{cfg.stage1.steps} steps"
    )

# Хотя мы не используем IDM для лейблов, загружаем его для совместимости
state_dicts["idm"] = clean_state_dict(state_dicts["idm"])
state_dicts["wm"] = clean_state_dict(state_dicts["wm"])
idm, _ = utils.create_dynamics_models(cfg.model, state_dicts=state_dicts)
idm.eval()

# Загружаем данные
train_data, test_data = data_loader.load(cfg.env_name)

# ИСПРАВЛЕНИЕ: Получаем размерность действий из одного батча данных
# Создаем временный итератор для получения размерности действий
temp_iter = train_data.get_iter(batch_size=1)
temp_batch = next(temp_iter)

# Получаем размерность действий из батча
if isinstance(temp_batch, dict) and "actions" in temp_batch:
    sample_actions = temp_batch["actions"]
    if sample_actions.dtype == torch.long:
        # Для дискретных действий нужно найти максимальное значение
        # Проверим несколько батчей для определения полного диапазона
        max_action = 0
        for _ in range(10):  # проверим 10 батчей
            batch = next(temp_iter)
            if isinstance(batch, dict) and "actions" in batch:
                batch_max = batch["actions"].max().item()
                max_action = max(max_action, batch_max)
        
        real_action_dim = max_action + 1  # +1 потому что действия начинаются с 0
    else:
        # Для непрерывных действий просто берем размерность
        real_action_dim = sample_actions.shape[-1]
else:
    # Fallback: используем конфигурацию окружения
    if hasattr(cfg, 'env') and hasattr(cfg.env, 'action_dim'):
        real_action_dim = cfg.env.action_dim
    else:
        # Для Procgen обычно 15 дискретных действий
        real_action_dim = 15
        doy.log(f"[yellow]Warning: Using default action dimension {real_action_dim} for Procgen")

print(f"Real action dimension: {real_action_dim}")
print(f"Latent action dimension: {cfg.model.la_dim}")

# Создаем политику с выходом для реальных действий
policy = utils.create_policy(cfg.model, real_action_dim)

# get gpu from command line
def get_arg(name, default):
    for arg in sys.argv:
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return default

gpu = int(get_arg("gpu", 0))
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

# Перемещаем модели на устройство
policy = policy.to(device)
idm = idm.to(device)

opt, lr_sched = doy.LMScheduler.make(
    policy=(
        doy.PiecewiseLinearSchedule(
            [0, 1000, cfg.stage2.steps + 1], [0.01 * cfg.stage2.lr, cfg.stage2.lr, 0]
        ),
        [policy],
    ),
)

train_iter = train_data.get_iter(cfg.stage2.bs)
test_iter = test_data.get_iter(128)

# Проверяем метрики IDM для понимания качества латентных представлений
_, eval_metrics = utils.eval_latent_repr(train_data, idm)
doy.log(f"IDM metrics (for reference): {eval_metrics}")

run, logger = config.wandb_init("lapo_stage2_real_actions", config.get_wandb_cfg(cfg))

for step in loop(
    cfg.stage2.steps + 1, desc="[green bold](stage-2) Training policy for REAL actions via BC"
):
    lr_sched.step(step)

    policy.train()
    batch = next(train_iter)
    
    # Move batch to device if it's a tensor or dict of tensors
    if isinstance(batch, dict):
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
    elif hasattr(batch, 'to'):
        batch = batch.to(device)
    
    # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Используем реальные действия вместо латентных
    # Получаем реальные действия из батча
    real_actions = batch["actions"]  # Используем реальные действия из данных
    
    # Политика предсказывает реальные действия на основе наблюдений
    obs = batch["obs"][:, -2]  # Берем последнее наблюдение перед переходом
    pred_actions = policy(obs)  # Политика предсказывает реальные действия
    
    # Вычисляем loss в зависимости от типа действий
    if real_actions.dtype == torch.long:
        # Дискретные действия - используем cross entropy
        loss = F.cross_entropy(pred_actions, real_actions)
    else:
        # Непрерывные действия - используем MSE
        loss = F.mse_loss(pred_actions, real_actions)

    opt.zero_grad()
    loss.backward()
    opt.step()

    logger(
        step=step,
        loss=loss,
        **lr_sched.get_state(),
    )

    if step % 200 == 0:
        policy.eval()
        test_batch = next(test_iter)
        
        # Move test_batch to device
        if isinstance(test_batch, dict):
            test_batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in test_batch.items()}
        elif hasattr(test_batch, 'to'):
            test_batch = test_batch.to(device)
        
        # Тестируем на реальных действиях
        test_real_actions = test_batch["actions"]
        test_obs = test_batch["obs"][:, -2]
        test_pred_actions = policy(test_obs)
        
        # Вычисляем тестовый loss
        if test_real_actions.dtype == torch.long:
            test_loss = F.cross_entropy(test_pred_actions, test_real_actions)
        else:
            test_loss = F.mse_loss(test_pred_actions, test_real_actions)
            
        logger(step=step, test_loss=test_loss)
        
        # Дополнительные метрики для дискретных действий
        if test_real_actions.dtype == torch.long:
            with torch.no_grad():
                pred_actions_discrete = torch.argmax(test_pred_actions, dim=-1)
                accuracy = (pred_actions_discrete == test_real_actions).float().mean()
                logger(step=step, test_accuracy=accuracy)
                
                # Выводим распределение предсказанных действий
                if step % 1000 == 0:
                    action_counts = torch.bincount(pred_actions_discrete, minlength=real_action_dim)
                    doy.log(f"Predicted action distribution: {action_counts.cpu().numpy()}")
                    
                    real_action_counts = torch.bincount(test_real_actions, minlength=real_action_dim)
                    doy.log(f"Real action distribution: {real_action_counts.cpu().numpy()}")

# Сохраняем обученную политику
torch.save(
    dict(
        policy=doy.state_dict_orig(policy), 
        cfg=cfg, 
        logger=logger,
        real_action_dim=real_action_dim,
        training_type="real_actions"
    ),
    paths.get_latent_policy_path(cfg.exp_name + "_real_actions"),
)

print(f"[bold green]Saved real action policy to {paths.get_latent_policy_path(cfg.exp_name + '_real_actions')}")

# Функция для тестирования обученной политики
def test_policy_on_real_actions(policy, test_data, device, num_batches=10):
    """Тестирует политику на реальных действиях"""
    policy.eval()
    total_loss = 0
    total_accuracy = 0
    total_samples = 0
    
    test_iter = test_data.get_iter(128)
    
    with torch.no_grad():
        for i in range(num_batches):
            batch = next(test_iter)
            if isinstance(batch, dict):
                batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            
            real_actions = batch["actions"]
            obs = batch["obs"][:, -2]
            pred_actions = policy(obs)
            
            if real_actions.dtype == torch.long:
                loss = F.cross_entropy(pred_actions, real_actions)
                pred_discrete = torch.argmax(pred_actions, dim=-1)
                accuracy = (pred_discrete == real_actions).float().mean()
                total_accuracy += accuracy.item() * real_actions.size(2)
            else:
                loss = F.mse_loss(pred_actions, real_actions)
                
            total_loss += loss.item() * real_actions.size(0)
            total_samples += real_actions.size(0)
    
    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples if total_accuracy > 0 else 0
    
    print(f"Test Results - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    return avg_loss, avg_accuracy

# Тестируем обученную политику
print("\n[bold blue]Testing trained policy on real actions:")
test_policy_on_real_actions(policy, test_data, device)