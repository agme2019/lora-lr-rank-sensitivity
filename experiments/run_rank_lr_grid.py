import argparse
import json
import numpy as np

from src.utils import set_seed, pick_device, ensure_dir
from src.data import make_splits, make_loaders
from src.sweeps import pretrain_base, run_rank_lr_grid
from src.plotting import plot_rank_lr_heatmaps, plot_best_of_lr_vs_rank

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda","mps","auto"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--finetune_epochs", type=int, default=10)
    ap.add_argument("--pretrain_epochs", type=int, default=10)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print("Using device:", device)

    pre_tr, pre_va, ft_tr, ft_va = make_splits(seed=args.seed)
    pre_tr_loader = make_loaders(pre_tr, batch_size=256, shuffle=True)
    pre_va_loader = make_loaders(pre_va, batch_size=256, shuffle=False)
    ft_tr_loader  = make_loaders(ft_tr,  batch_size=256, shuffle=True)
    ft_va_loader  = make_loaders(ft_va,  batch_size=256, shuffle=False)

    print("Pretraining base...")
    state, pre_acc = pretrain_base(
        device=device,
        pre_tr_loader=pre_tr_loader,
        pre_va_loader=pre_va_loader,
        seq_len=32, vocab=64, num_classes=64,
        lr=3e-4,
        epochs=args.pretrain_epochs,
    )
    print("Pretrain val acc:", pre_acc)

    lr_grid = np.logspace(-6, -2.2, 16)

    grid = run_rank_lr_grid(
        device=device,
        state_dict=state,
        ft_tr_loader=ft_tr_loader,
        ft_va_loader=ft_va_loader,
        seq_len=32, vocab=64, num_classes=64,
        ranks=args.ranks,
        lr_grid=lr_grid,
        finetune_epochs=args.finetune_epochs,
    )

    ensure_dir("artifacts")
    with open("artifacts/rank_x_lr_results.json", "w") as f:
        json.dump(grid, f, indent=2)

    paths = plot_rank_lr_heatmaps(grid, out_dir="figures")
    best = plot_best_of_lr_vs_rank(grid, out_dir="figures")
    print("Saved:", best)
    for p in paths:
        print("Saved:", p)
    print("Saved: artifacts/rank_x_lr_results.json")

if __name__ == "__main__":
    main()
