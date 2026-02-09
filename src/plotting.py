import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import ensure_dir

def plot_lr_sweep(results, out_dir="figures"):
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "raw"))

    plt.figure(figsize=(8.5, 5))
    for kind, v in results.items():
        lrs = np.array(v["lrs"], dtype=float)
        acc = np.array(v["acc"], dtype=float)
        order = np.argsort(lrs)
        plt.plot(lrs[order], acc[order], marker="o", label=v.get("name", kind))
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Validation accuracy")
    plt.title("LR sweep parity (synthetic)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "raw", "lr_sweep_parity.png")
    plt.savefig(path, dpi=220)
    plt.close()
    return path

def _matrix_from_rank_grid(rank_entry):
    lrs = np.array(rank_entry["lrs"], dtype=float)
    return lrs

def plot_rank_lr_heatmaps(results_rank, out_dir="figures"):
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "raw"))
    ensure_dir(os.path.join(out_dir, "normalized"))

    saved = []
    for kind, method in results_rank.items():
        ranks = sorted(int(r) for r in method["ranks"].keys())
        lrs = np.array(method["ranks"][str(ranks[0])]["lrs"], dtype=float)
        logLR = np.log10(lrs)

        # raw matrix
        M = np.array([method["ranks"][str(r)]["acc"] for r in ranks], dtype=float)

        # raw heatmap
        plt.figure(figsize=(9, 4.8))
        im = plt.imshow(
            M,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[logLR.min(), logLR.max(), min(ranks), max(ranks)],
        )
        plt.colorbar(im, label="Validation accuracy")
        plt.xlabel("log10(LR)")
        plt.ylabel("Rank r")
        plt.title(f"Rank × LR heatmap — {method.get('name', kind)}")
        plt.tight_layout()
        raw_path = os.path.join(out_dir, "raw", f"heatmap_{kind}.png")
        plt.savefig(raw_path, dpi=220)
        plt.close()
        saved.append(raw_path)

        # normalized per rank
        Mn = M / (M.max(axis=1, keepdims=True) + 1e-12)

        plt.figure(figsize=(9, 4.8))
        im = plt.imshow(
            Mn,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[logLR.min(), logLR.max(), min(ranks), max(ranks)],
            vmin=0.0,
            vmax=1.0,
        )
        cbar = plt.colorbar(im)
        cbar.set_label("Relative performance (per-rank max = 1.0)")
        plt.xlabel("log10(LR)")
        plt.ylabel("Rank r")
        plt.title(f"Normalized Rank × LR sensitivity — {method.get('name', kind)}")
        plt.tight_layout()
        norm_path = os.path.join(out_dir, "normalized", f"heatmap_normalized_{kind}.png")
        plt.savefig(norm_path, dpi=220)
        plt.close()
        saved.append(norm_path)

    return saved

def plot_best_of_lr_vs_rank(results_rank, out_dir="figures"):
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "raw"))

    plt.figure(figsize=(9, 4.8))
    for kind, method in results_rank.items():
        ranks = sorted(int(r) for r in method["ranks"].keys())
        bests = [method["ranks"][str(r)]["best"] for r in ranks]
        plt.plot(ranks, bests, marker="o", label=method.get("name", kind))

    plt.xlabel("Rank r")
    plt.ylabel("Best val accuracy (over LR sweep)")
    plt.title("Best-of-LR vs rank")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "raw", "best_of_lr_vs_rank.png")
    plt.savefig(path, dpi=220)
    plt.close()
    return path
