from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

from models.base import BaseModel, VAEOutput


@dataclass
class EmbeddingBetaVAEConfig:
    input_dim: int
    gamma: int = 1000
    C_start: int = 0
    C_max: int = 25
    C_iters_to_max: int = 50000


class BetaVAEForEmbedding(BaseModel):
    """
    Limited-capacity Beta VAE for dense embedding vectors.
    """

    def __init__(self, config: EmbeddingBetaVAEConfig) -> None:
        super().__init__(config=config)
        self.global_step = 0
        self.rec_loss = mse_loss

        self.encoder = nn.Sequential(
            nn.Linear(self.config.input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.config.input_dim),
        )

        self.mu_proj = nn.Linear(256, 64)
        self.logvar_proj = nn.Linear(256, 64)

    def _capacity_schedule(self) -> torch.Tensor:
        if self.config.C_iters_to_max <= 0:
            return torch.tensor(float(self.config.C_max))

        t = min(self.global_step / self.config.C_iters_to_max, 1.0)
        capacity = self.config.C_start + t * (self.config.C_max - self.config.C_start)
        return torch.tensor(float(capacity))

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode embedding vectors into Gaussian latent parameters.

        Args:
            x (torch.Tensor): Embedding batch with shape ``(batch_size, input_dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and log-variance tensors.
        """
        h = self.encoder(x)
        return self.mu_proj(h), self.logvar_proj(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors into reconstructed embeddings.

        Args:
            z (torch.Tensor): Latent batch with shape ``(batch_size, 64)``.

        Returns:
            torch.Tensor: Reconstructed embedding batch.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> VAEOutput:
        """
        Run a limited-capacity Beta VAE forward pass.

        Args:
            x (torch.Tensor): Embedding batch with shape ``(batch_size, input_dim)``.

        Returns:
            VAEOutput: Reconstruction and VAE loss bundle.
        """
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        reconstructed = self.decode(z)

        capacity = self._capacity_schedule().to(device=mu.device)
        if self.training:
            self.global_step += 1

        rec_loss = self.rec_loss(reconstructed, x, reduction='none').sum(dim=-1)
        kld_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        loss = rec_loss + self.config.gamma * torch.abs(kld_loss - capacity)

        return VAEOutput(
            rec=reconstructed,
            loss=loss.mean(),
            rec_loss=rec_loss.mean(),
            kld_loss=kld_loss.mean(),
        )
