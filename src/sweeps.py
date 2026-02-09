import json
import numpy as np
import torch

from .model import TinyTransformer
from .adapters import inject_adapter, unfreeze_adapters_and_head
from .train import train_epochs, eval_acc
from .utils import ensure_dir

def pretrain_base(
    device: torch.device,
    pre_tr_loader,
    pre_va_loader,
    seq_len: int,
    vocab: int,
    num_classes: int,
    d_model: int = 192,
    n_heads: int = 4,
    n_layers: int = 2,
    lr: float = 3e-4,
    epochs: int = 10,
    seed_state_out: str = None,
):
    model = TinyTransformer(vocab=vocab, seq_len=seq_len, d_model=d_model, n_heads=n_heads, n_layers=n_layers, num_classes=num_classes)
    model = model.to(device)

    # train all params
    for p in model.parameters():
        p.requires_grad = True

    train_epochs(model, pre_tr_loader, device=device, lr=lr, epochs=epochs, weight_decay=0.0)
    acc = eval_acc(model, pre_va_loader, device=device)

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if seed_state_out is not None:
        ensure_dir("artifacts")
        torch.save(state, seed_state_out)

    return state, acc

def run_lr_sweep(
    device: torch.device,
    state_dict,
    ft_tr_loader,
    ft_va_loader,
    seq_len: int,
    vocab: int,
    num_classes: int,
    method_kinds=("lora","dora","pissa","milora"),
    rank: int = 16,
    lr_grid=None,
    finetune_epochs: int = 10,
):
    if lr_grid is None:
        lr_grid = np.logspace(-6, -2.2, 16)

    results = {}
    for kind in method_kinds:
        results[kind] = {"name": kind, "lrs": [], "acc": []}

        for lr in lr_grid:
            model = TinyTransformer(vocab=vocab, seq_len=seq_len, num_classes=num_classes)
            model.load_state_dict(state_dict, strict=True)

            # inject BEFORE .to(device)
            inject_adapter(model, kind=kind, r=rank, skip_names=("head",))
            model = model.to(device)
            unfreeze_adapters_and_head(model)

            train_epochs(model, ft_tr_loader, device=device, lr=float(lr), epochs=finetune_epochs, weight_decay=0.0)
            acc = eval_acc(model, ft_va_loader, device=device)

            results[kind]["lrs"].append(float(lr))
            results[kind]["acc"].append(float(acc))

    return results

def run_rank_lr_grid(
    device: torch.device,
    state_dict,
    ft_tr_loader,
    ft_va_loader,
    seq_len: int,
    vocab: int,
    num_classes: int,
    method_kinds=("lora","dora","pissa","milora"),
    ranks=(4,8,16,32),
    lr_grid=None,
    finetune_epochs: int = 10,
):
    if lr_grid is None:
        lr_grid = np.logspace(-6, -2.2, 16)

    out = {}
    for kind in method_kinds:
        out[kind] = {"name": kind, "ranks": {}}
        for r in ranks:
            accs = []
            for lr in lr_grid:
                model = TinyTransformer(vocab=vocab, seq_len=seq_len, num_classes=num_classes)
                model.load_state_dict(state_dict, strict=True)

                inject_adapter(model, kind=kind, r=int(r), skip_names=("head",))
                model = model.to(device)
                unfreeze_adapters_and_head(model)

                train_epochs(model, ft_tr_loader, device=device, lr=float(lr), epochs=finetune_epochs, weight_decay=0.0)
                acc = eval_acc(model, ft_va_loader, device=device)
                accs.append(float(acc))

            best = float(np.max(accs))
            best_lr = float(lr_grid[int(np.argmax(accs))])

            out[kind]["ranks"][str(r)] = {
                "lrs": [float(x) for x in lr_grid],
                "acc": accs,
                "best": best,
                "best_lr": best_lr,
            }
    return out
