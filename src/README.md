````markdown
# How to Use This Repository

This repository provides a **fully reproducible pipeline** for studying **learning-rate × rank sensitivity** in LoRA-style adapter methods (LoRA, DoRA, PiSSA-like, MiLoRA-like).

You can run everything **without opening a notebook**.  
The notebook is optional and intended only for exploration.

Repo:  
https://github.com/agme2019/lora-lr-rank-sensitivity

---

## 1. Setup

### Clone the repository
```bash
git clone https://github.com/agme2019/lora-lr-rank-sensitivity.git
cd lora-lr-rank-sensitivity
````

### Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Device selection (important)

By default, **everything runs on CPU**.

You can optionally use acceleration if available:

* `--device auto` → CUDA (if available) → MPS (Apple Silicon) → CPU
* `--device cuda`
* `--device mps`

This is handled automatically and safely.

---

## 3. Recommended way to run experiments (scripts)

### A. Learning-rate sweep (single rank)

This reproduces the **LR sweep parity** plots.

```bash
python experiments/run_lr_sweep.py --device cpu
```

or, with acceleration:

```bash
python experiments/run_lr_sweep.py --device auto
```

**Outputs**

* `artifacts/lr_sweep_results.json`
* `figures/raw/lr_sweep_parity.png`

---

### B. Rank × learning-rate grid (main experiment)

This reproduces **all heatmaps**, including the **normalized “killer” figure**.

```bash
python experiments/run_rank_lr_grid.py --device auto
```

**Outputs**

* `artifacts/rank_x_lr_results.json`
* `figures/raw/heatmap_*.png`
* `figures/normalized/heatmap_normalized_*.png`
* `figures/raw/best_of_lr_vs_rank.png`

This is the **core experiment** used in the README/blog discussion.

---

## 4. Where results are saved

```
artifacts/
  ├── lr_sweep_results.json
  └── rank_x_lr_results.json

figures/
  ├── raw/
  │   ├── lr_sweep_parity.png
  │   ├── heatmap_lora.png
  │   ├── heatmap_dora.png
  │   ├── heatmap_pissa.png
  │   ├── heatmap_milora.png
  │   └── best_of_lr_vs_rank.png
  └── normalized/
      ├── heatmap_normalized_lora.png
      ├── heatmap_normalized_dora.png
      ├── heatmap_normalized_pissa.png
      └── heatmap_normalized_milora.png
```

---

## 5. Notebook usage (optional)

The notebook in `notebooks/` is **not required** to reproduce results.

Use it if you want to:

* inspect intermediate tensors
* tweak hyperparameters interactively
* visualize results step-by-step
* debug adapter behavior

The notebook **imports functions from `src/`**, so there is a single source of truth.

> If you can run the scripts, you do not need the notebook.

---

## 6. Modifying experiments

Common knobs to change (in scripts):

* **Ranks**

  ```bash
  --ranks 4 8 16 32
  ```

* **Finetuning epochs**

  ```bash
  --finetune_epochs 15
  ```

* **Pretraining epochs**

  ```bash
  --pretrain_epochs 20
  ```

All experiments remain reproducible.

---

## 7. What this repo is (and isn’t)

**This repo is:**

* a diagnostic study of LR sensitivity
* a controlled synthetic benchmark
* a companion to MiLoRA-style analysis

**This repo is not:**

* a leaderboard benchmark
* a production fine-tuning recipe
* a claim of new adapter methods

---

## 8. Reference

This work is inspired by:

> **Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning**
> arXiv:2602.04998
> [https://arxiv.org/abs/2602.04998](https://arxiv.org/abs/2602.04998)

---

## 9. Quick sanity check

If everything worked correctly, you should see:

* Different optimal LRs for different methods
* Strong rank–LR interaction
* After per-rank normalization, **very similar LR sensitivity across methods**


