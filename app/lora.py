"""LoRA (Low-Rank Adaptation), from scratch.

A frozen weight W gets a trainable low-rank update:  W' = W + (alpha / r) * B @ A
with A: (r, in) and B: (out, r). B starts at zero, so at injection time the model is exactly the
pretrained one. Only A and B train, so a layer with in*out weights trains just r*(in+out).

GPT-2 uses `transformers.pytorch_utils.Conv1D`, not nn.Linear, for its projections. Conv1D stores its
weight transposed, as (in, out), and computes x @ W + b. Both layouts are handled here.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D


def _in_out(layer: nn.Module) -> tuple[int, int]:
    if isinstance(layer, Conv1D):
        return layer.weight.shape[0], layer.weight.shape[1]
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    raise TypeError(f"LoRA supports Conv1D and nn.Linear, not {type(layer).__name__}")


class LoRALayer(nn.Module):
    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r < 1:
            raise ValueError("LoRA rank r must be >= 1")
        self.base = base
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        n_in, n_out = _in_out(base)
        self.lora_A = nn.Parameter(torch.empty(r, n_in))
        self.lora_B = nn.Parameter(torch.zeros(n_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return self.base(x) + self.scale * update

    def delta_weight(self) -> torch.Tensor:
        """The update to add to the base weight, in the base layer's own layout."""
        delta = self.scale * (self.lora_B @ self.lora_A)  # (out, in)
        return delta.T if isinstance(self.base, Conv1D) else delta

    def merged(self) -> nn.Module:
        """Fold the update into the base weight and return the plain layer (no extra cost at inference)."""
        with torch.no_grad():
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
        return self.base


def _set_child(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parent_name, _, child = dotted.rpartition(".")
    setattr(root.get_submodule(parent_name) if parent_name else root, child, new)


def inject_lora(model: nn.Module, targets=("attn.c_attn",), r: int = 8, alpha: float = 16.0,
                dropout: float = 0.0) -> list[str]:
    """Freeze every weight in `model`, then wrap each layer whose name ends with one of `targets`.

    GPT-2 names: attn.c_attn (fused Q,K,V), attn.c_proj (attention output), mlp.c_fc, mlp.c_proj.
    Returns the names of the wrapped layers.
    """
    for p in model.parameters():
        p.requires_grad = False
    names = [name for name, module in model.named_modules()
             if isinstance(module, (Conv1D, nn.Linear)) and any(name.endswith(t) for t in targets)]
    if not names:
        raise ValueError(f"no layers match {targets}")
    for name in names:
        _set_child(model, name, LoRALayer(model.get_submodule(name), r=r, alpha=alpha, dropout=dropout))
    return names


def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALayer with its merged base layer, in place. Returns how many were merged."""
    names = [name for name, module in model.named_modules() if isinstance(module, LoRALayer)]
    for name in names:
        _set_child(model, name, model.get_submodule(name).merged())
    return len(names)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
