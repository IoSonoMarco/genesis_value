from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AEModelConfig:
    latent_dim: int
    hidden_dim_base: int
    m: int = 3
    kld_beta: float | None = None # only for VAE
    cond_channels: int | None = None # only for FiLM
    
    def __post_init__(self):
        if self.kld_beta is None:
            self.kld_beta = 1.0


@dataclass
class LimitedCapacityAEModelConfig(AEModelConfig):
    gamma: int = 1000
    C_start: int = 0
    C_max: int = 25
    C_iters_to_max: int = 50000


@dataclass
class VAEOutput:
    rec: torch.Tensor
    loss: torch.Tensor
    rec_loss: torch.Tensor
    kld_loss: torch.Tensor
    reward: torch.Tensor | None = None
    reward_loss: torch.Tensor | None = None
    raw_reward_mse: torch.Tensor | None = None

    def __post_init__(self):
        self.rec_loss = self.rec_loss.detach()
        self.kld_loss = self.kld_loss.detach()
        if self.reward_loss is not None:
            self.reward_loss = self.reward_loss.detach()
        if self.raw_reward_mse is not None:
            self.raw_reward_mse = self.raw_reward_mse.detach()


class BaseModel(nn.Module):
    def __init__(self, config: AEModelConfig):
        super().__init__()
        self.config = config

    def show_number_of_parameters(self):
        tot_params = sum(p.numel() for p in self.parameters())
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        tot_params = round(tot_params / 1e+6, 3)
        train_params = round(train_params / 1e+6, 3)
        print(f"{train_params}M / {tot_params}M")
