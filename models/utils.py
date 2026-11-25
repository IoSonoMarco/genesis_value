import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SkipConnectionWrapper(nn.Module):
    """
    Add residual connection to a Module. If in_channels != out_channels,
    automatically applies a 1×1 Conv1d to the input for matching dimensions.
    """
    def __init__(self, module: nn.Module, in_channels: int, out_channels: int):
        super().__init__()
        self.module = module
        self.matching_projection = nn.Identity()
        if in_channels != out_channels:
            self.matching_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        res = self.matching_projection(x)
        return out + res
    

class FiLM2d(nn.Module):
    """
    Spatial FiLM: produce per-(C,H,W) gamma/beta from a conditioning map.
    Keeps identity at init: y = (1+gamma)*x + beta with gamma=0, beta=0.
    """
    def __init__(self, cond_channels: int, out_channels: int, hidden: int = 0):
        super().__init__()
        h = hidden or max(32, cond_channels)
        self.adapter = nn.Sequential(
            nn.Conv2d(cond_channels, h, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(h, 2 * out_channels, kernel_size=1)
        )
        # Identity init
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Resize cond to match x spatially
        c = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gamma_beta = self.adapter(c)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return x * (1 + gamma) + beta
    
###

def softmax(x, temperature=1.0):
    logits = x / temperature
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()

def calibrated_softmax(scores, temperature=1.0):
    if scores.size == 0:
        return scores
    
    std = scores.std()
    if std == 0.0:
        return np.full(scores.shape, 1.0 / scores.size)
    z_scores = (scores - scores.mean()) / std
    
    return softmax(z_scores, temperature)
