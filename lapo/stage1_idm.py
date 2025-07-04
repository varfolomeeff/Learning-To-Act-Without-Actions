import config
import data_loader
import doy
import paths
import torch
import utils
from doy import loop
import torch.nn as nn

cfg = config.get()
doy.print("[bold green]Running LAPO stage 1 (IDM/FDM training) with config:")
config.print_cfg(cfg)

run, logger = config.wandb_init("lapo_stage1", config.get_wandb_cfg(cfg))

idm, wm = utils.create_dynamics_models(cfg.model)

# Multi-GPU support
if torch.cuda.is_available():
    device = torch.device('cuda')
    idm = idm.to(device)
    wm = wm.to(device)
    if torch.cuda.device_count() > 1:
        idm = torch.nn.DataParallel(idm)
        wm = torch.nn.DataParallel(wm)
    print("CUDA device count:", torch.cuda.device_count())
else:
    device = torch.device('cpu')

print("Using DataParallel:", isinstance(idm, torch.nn.DataParallel))
print("CUDA device count:", torch.cuda.device_count())

train_data, test_data = data_loader.load(cfg.env_name)
train_iter = train_data.get_iter(cfg.stage1.bs)
test_iter = test_data.get_iter(128)

opt, lr_sched = doy.LRScheduler.make(
    all=(
        doy.PiecewiseLinearSchedule(
            [0, 50, cfg.stage1.steps + 1],
            [0.1 * cfg.stage1.lr, cfg.stage1.lr, 0.01 * cfg.stage1.lr],
        ),
        [wm, idm],
    ),
)

def print_device(self, *args, **kwargs):
    print(f"Running on device: {next(self.parameters()).device}")
    return self._original_forward(*args, **kwargs)

idm._original_forward = idm.forward
idm.forward = print_device.__get__(idm, type(idm))

def train_step():
    idm.train()
    wm.train()

    lr_sched.step(step)

    batch = next(train_iter)
    # Move batch to device if it's a tensor or dict of tensors
    if isinstance(batch, dict):
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
    elif hasattr(batch, 'to'):
        batch = batch.to(device)

    # Use .module if model is DataParallel
    idm_label = idm.module.label if isinstance(idm, torch.nn.DataParallel) else idm.label
    wm_label = wm.module.label if isinstance(wm, torch.nn.DataParallel) else wm.label
    vq_loss, vq_perp = idm_label(batch)
    wm_loss = wm_label(batch)
    loss = wm_loss + vq_loss

    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([*idm.parameters(), *wm.parameters()], 2)
    opt.step()

    logger(
        step,
        wm_loss=wm_loss,
        global_step=step * cfg.stage1.bs,
        vq_perp=vq_perp,
        vq_loss=vq_loss,
        grad_norm=grad_norm,
        **lr_sched.get_state(),
    )


def test_step():
    idm.eval()  # disables idm.vq ema update
    wm.eval()

    # evaluate IDM + FDM generalization on (action-free) test data
    batch = next(test_iter)
    # Move batch to device if it's a tensor or dict of tensors
    if isinstance(batch, dict):
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
    elif hasattr(batch, 'to'):
        batch = batch.to(device)
    idm_label = idm.module.label if isinstance(idm, torch.nn.DataParallel) else idm.label
    wm_label = wm.module.label if isinstance(wm, torch.nn.DataParallel) else wm.label
    idm_label(batch)
    wm_loss = wm_label(batch)

    # train latent -> true action decoder and evaluate its predictiveness
    _, eval_metrics = utils.eval_latent_repr(train_data, idm)

    logger(step, wm_loss_test=wm_loss, global_step=step * cfg.stage1.bs, **eval_metrics)


for step in loop(cfg.stage1.steps + 1, desc="[green bold](stage-1) Training IDM + FDM"):
    train_step()

    if step % 500 == 0:
        test_step()

    if step > 0 and (step % 5_000 == 0 or step == cfg.stage1.steps):
        torch.save(
            dict(
                **doy.get_state_dicts(wm=wm, idm=idm, opt=opt),
                step=step,
                cfg=cfg,
                logger=logger,
            ),
            paths.get_models_path(cfg.exp_name),
        )

import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def forward(self, x):
        print(f"Device in forward: {x.device}, id: {torch.cuda.current_device()}")
        return x.sum()

model = ToyModel().cuda()
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)
x = torch.randn(32, 3, 224, 224).cuda()
model(x)
