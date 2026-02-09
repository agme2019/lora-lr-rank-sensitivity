import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RetrieveAtK(Dataset):
    """
    Synthetic classification: input is a sequence of token ids (ints).
    Target is a class derived from a hidden rule.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def _make_rule_labels(X: np.ndarray, vocab: int, num_classes: int, k: int, seed: int = 0):
    """
    Hidden rule: pick k token positions, hash them with random weights -> class.
    """
    rng = np.random.default_rng(seed)
    seq_len = X.shape[1]
    idx = rng.choice(seq_len, size=k, replace=False)
    w = rng.integers(1, 97, size=k, endpoint=True)
    s = (X[:, idx] * w).sum(axis=1)
    return (s % num_classes).astype(np.int64)

def make_splits(
    n_pretrain: int = 6000,
    n_finetune: int = 3000,
    seq_len: int = 32,
    vocab: int = 64,
    num_classes: int = 64,
    k_pre: int = 7,
    k_ft: int = 31,
    seed: int = 0,
):
    """
    Returns:
      pre_tr, pre_va, ft_tr, ft_va as Datasets
    """
    rng = np.random.default_rng(seed)

    # Pretrain distribution
    X_pre = rng.integers(0, vocab, size=(n_pretrain, seq_len), endpoint=False)
    y_pre = _make_rule_labels(X_pre, vocab, num_classes, k=k_pre, seed=seed + 1)

    # Finetune distribution (harder / different rule)
    X_ft = rng.integers(0, vocab, size=(n_finetune, seq_len), endpoint=False)
    y_ft = _make_rule_labels(X_ft, vocab, num_classes, k=k_ft, seed=seed + 2)

    # Split 80/20
    def split(X, y):
        n = X.shape[0]
        n_tr = int(0.8 * n)
        return (X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:])

    Xp_tr, yp_tr, Xp_va, yp_va = split(X_pre, y_pre)
    Xf_tr, yf_tr, Xf_va, yf_va = split(X_ft, y_ft)

    pre_tr = RetrieveAtK(Xp_tr, yp_tr)
    pre_va = RetrieveAtK(Xp_va, yp_va)
    ft_tr  = RetrieveAtK(Xf_tr, yf_tr)
    ft_va  = RetrieveAtK(Xf_va, yf_va)

    return pre_tr, pre_va, ft_tr, ft_va

def make_loaders(dataset, batch_size: int = 128, shuffle: bool = True, num_workers: int = 0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
