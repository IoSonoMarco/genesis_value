from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm


class LinearDecoder(nn.Module):
    """
    Project scalar values into latent embeddings and reconstruct them linearly.

    Args:
        hidden_dim (int): Hidden state width after scalar projection.
        latent_dim (int): Latent embedding width.
    """

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(1, hidden_dim)
        self.activation = nn.ReLU()
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, 1)
        self.rec_loss = nn.MSELoss()
        self.dataloader: DataLoader

    @staticmethod
    def _as_column(scalars: torch.Tensor) -> torch.Tensor:
        if scalars.ndim == 1:
            return scalars.unsqueeze(-1)
        return scalars

    def encode(self, scalars: torch.Tensor) -> torch.Tensor:
        """
        Encode scalar values into latent embeddings.

        Args:
            scalars (torch.Tensor): Scalar tensor with shape ``[B]`` or ``[B, 1]``.

        Returns:
            torch.Tensor: Latent embeddings with shape ``[B, latent_dim]``.
        """
        hidden = self.activation(self.input_proj(self._as_column(scalars)))
        return self.latent_proj(hidden)

    def decode(self, latent_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode latent embeddings into scalar reconstructions.

        Args:
            latent_embeddings (torch.Tensor): Latent embeddings with shape ``[B, latent_dim]``.

        Returns:
            torch.Tensor: Reconstructed scalars with shape ``[B, 1]``.
        """
        return self.decoder(latent_embeddings)

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct scalar values through the latent embedding.

        Args:
            scalars (torch.Tensor): Scalar tensor with shape ``[B]`` or ``[B, 1]``.

        Returns:
            torch.Tensor: Reconstructed scalars with shape ``[B, 1]``.
        """
        return self.decode(self.encode(scalars))

    def set_dataloader(self, scope: np.ndarray, batch_size: int) -> None:
        self.dataloader = DataLoader(
            TensorDataset(torch.tensor(scope)),
            batch_size=batch_size,
            shuffle=True,
        )

    def train_model(
        self,
        *,
        epochs: int,
        lr: float,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
    ) -> list[float]:
        """
        Train the scalar reconstruction objective.

        Args:
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            optimizer_cls (type[torch.optim.Optimizer]): Optimizer class.

        Returns:
            list[float]: Mean reconstruction loss for each epoch.
        """
        device = next(self.parameters()).device
        optimizer = optimizer_cls(self.parameters(), lr=lr)
        history = []

        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            sample_count = 0
            progress = tqdm(
                self.dataloader,
                desc=f'LinearDecoder | Epoch {epoch + 1}/{epochs}',
            )

            for batch in progress:
                scalars = self._as_column(batch[0]).float().to(device)
                reconstructed = self(scalars)
                loss = self.rec_loss(reconstructed, scalars)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach()) * scalars.shape[0]
                sample_count += scalars.shape[0]
                progress.set_postfix({'loss': total_loss / sample_count})

            history.append(total_loss / sample_count)

        return history


def load_linear_decoder(
    *,
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> LinearDecoder:
    """
    Load a trained linear decoder from a weights-only checkpoint.

    Args:
        checkpoint_path (str | Path): Path to a saved ``LinearDecoder.state_dict()``.
        device (torch.device | str): Device for inference.

    Returns:
        LinearDecoder: Model with loaded weights in eval mode.
    """
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    model = LinearDecoder(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
