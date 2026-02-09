import torch
import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def eval_acc(model, loader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

def train_epochs(model, loader, device: torch.device, lr: float, epochs: int = 10, weight_decay: float = 0.0):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
