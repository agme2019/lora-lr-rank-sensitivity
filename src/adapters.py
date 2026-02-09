import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.A = nn.Linear(linear.in_features, r, bias=False)
        self.B = nn.Linear(r, linear.out_features, bias=False)

        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        base = self.linear(x)
        delta = self.B(self.A(x)) * self.scaling
        return base + delta

class DoRALinear(nn.Module):
    """
    Simplified DoRA: decouple direction + magnitude.
    """
    def __init__(self, linear: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.A = nn.Linear(linear.in_features, r, bias=False)
        self.B = nn.Linear(r, linear.out_features, bias=False)
        self.m = nn.Parameter(torch.ones(linear.out_features))

        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        W = self.linear.weight
        deltaW = (self.B.weight @ self.A.weight) * self.scaling
        W_eff = W + deltaW
        W_dir = F.normalize(W_eff, dim=1)
        y = x @ (W_dir * self.m.unsqueeze(1)).t()
        if self.linear.bias is not None:
            y = y + self.linear.bias
        return y

def _svd_init_lora(wrapper: LoRALinear, mode: str = "top"):
    """
    PiSSA-like: top-r SVD init
    MiLoRA-like: bottom-r SVD init
    """
    W = wrapper.linear.weight.detach().float().cpu()  # [out, in]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = wrapper.r
    if r > min(W.shape):
        raise ValueError(f"rank r={r} too large for weight shape {tuple(W.shape)}")

    if mode == "top":
        idx = torch.arange(r)
    elif mode == "bottom":
        idx = torch.arange(len(S) - r, len(S))
    else:
        raise ValueError("mode must be 'top' or 'bottom'")

    U_r = U[:, idx]         # [out, r]
    S_r = S[idx]            # [r]
    V_r = Vh[idx, :]        # [r, in]
    sqrtS = torch.sqrt(S_r)

    A = (sqrtS.unsqueeze(1) * V_r)          # [r, in]
    B = (U_r * sqrtS.unsqueeze(0))          # [out, r]

    with torch.no_grad():
        wrapper.A.weight.copy_(A.to(wrapper.A.weight.device))
        wrapper.B.weight.copy_(B.to(wrapper.B.weight.device))

def inject_adapter(model: nn.Module, kind: str, r: int, skip_names=("head",)):
    """
    Safe two-pass replacement (avoids recursion / iterator mutation).
    IMPORTANT: call this BEFORE model.to(device) for MPS safety.
    """
    kind = kind.lower()
    assert kind in ("lora", "dora", "pissa", "milora")

    targets = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if name in skip_names:
                continue
            if isinstance(child, nn.Linear):
                if isinstance(child, (LoRALinear, DoRALinear)):
                    continue
                targets.append((parent, name, child))

    for parent, name, child in targets:
        if kind == "lora":
            wrapped = LoRALinear(child, r=r, alpha=r)
        elif kind == "dora":
            wrapped = DoRALinear(child, r=r, alpha=r)
        elif kind == "pissa":
            wrapped = LoRALinear(child, r=r, alpha=r)
            _svd_init_lora(wrapped, mode="top")
        elif kind == "milora":
            wrapped = LoRALinear(child, r=r, alpha=r)
            _svd_init_lora(wrapped, mode="bottom")

        setattr(parent, name, wrapped)

    return model

def unfreeze_adapters_and_head(model: nn.Module):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze adapter modules
    for m in model.modules():
        if isinstance(m, (LoRALinear, DoRALinear)):
            for p in m.parameters():
                p.requires_grad = True

    # unfreeze head
    for p in model.head.parameters():
        p.requires_grad = True
