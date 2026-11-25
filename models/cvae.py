from dataclasses import replace
from typing import Literal

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits

from models.base import AEModelConfig, BaseModel, LimitedCapacityAEModelConfig, VAEOutput
from models.decoders import DecoderUpMFiLM
from models.encoders import EncoderDownM


class _BaseCVAEMultipleFeatures(BaseModel):
    """
    Base CVAE machinery for models conditioned on multiple categorical features.

    This class owns the VAE encoder path, latent sampling, feature embeddings, and VAE loss. Decoder-specific
    conditioning belongs to subclasses.
    """

    def __init__(
            self,
            config: AEModelConfig,
            feature_sizes: dict[str, int],
            n_cond_channels: int,
            image_size: int,
            reduction: Literal['mean', 'sum'] = 'mean',
    ) -> None:
        """
        Initialize shared CVAE components.

        Args:
            config (AEModelConfig): Model dimension and VAE loss configuration.
            feature_sizes (dict[str, int]): Categorical cardinality for each conditioning feature.
            n_cond_channels (int): Number of condition channels per feature.
            image_size (int): Square input image size.
            reduction (Literal['mean', 'sum']): Reconstruction loss reduction mode.

        Raises:
            ValueError: If reduction is not supported.
        """
        super().__init__(config=config)
        if reduction not in {'mean', 'sum'}:
            raise ValueError('reduction must be either \'mean\' or \'sum\'.')

        self.latent_dim = config.latent_dim
        self.hidden_dim_base = config.hidden_dim_base
        self.m = config.m
        self.kld_beta = config.kld_beta
        self.feature_sizes = dict(feature_sizes)
        self.feature_names = tuple(feature_sizes.keys())
        self.n_cond_channels = n_cond_channels
        self.image_size = image_size
        self.reduction = reduction
        self.rec_loss = binary_cross_entropy_with_logits

        self.image_encoder = EncoderDownM(config)

        with torch.no_grad():
            h = self.image_encoder(torch.randn(1, 3, image_size, image_size))
            self.hidden_shape = h.shape[1:]
            self.cond_shape = (
                n_cond_channels,
                self.hidden_shape[-2],
                self.hidden_shape[-1],
            )
            self.cond_dim = int(n_cond_channels * self.hidden_shape[-2] * self.hidden_shape[-1])

        self.feature_embeddings = nn.ModuleDict({
            name: nn.Embedding(size, self.cond_dim)
            for name, size in self.feature_sizes.items()
        })

        self.mu_proj = nn.Conv2d(self.latent_dim, self.latent_dim, 1)
        self.logvar_proj = nn.Conv2d(self.latent_dim, self.latent_dim, 1)

    def encode(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images into latent distribution parameters.

        Args:
            pixel_values (torch.Tensor): Batched input images.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and log-variance tensors.
        """
        h = self.image_encoder(pixel_values)
        return self.mu_proj(h), self.logvar_proj(h)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def embed_features(self, feature_ids: dict[str, torch.LongTensor]) -> torch.Tensor:
        """
        Embed feature IDs into a concatenated spatial condition tensor.

        Args:
            feature_ids (dict[str, torch.LongTensor]): Batched categorical IDs by feature name.

        Returns:
            torch.Tensor: Concatenated condition tensor with one block per configured feature.
        """
        parts = []
        for name in self.feature_names:
            ids = feature_ids[name]
            embedding = self.feature_embeddings[name](ids)
            parts.append(embedding.view(ids.shape[0], *self.cond_shape))
        return torch.cat(parts, dim=1)

    def vae_loss(
            self,
            reconstructed: torch.Tensor,
            pixel_values: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the CVAE reconstruction and KL losses.

        Args:
            reconstructed (torch.Tensor): Decoder image logits.
            pixel_values (torch.Tensor): Target image values.
            mu (torch.Tensor): Latent mean tensor.
            logvar (torch.Tensor): Latent log-variance tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Total, reconstruction, and KL losses.
        """
        if self.reduction == 'mean':
            rec_loss = self.rec_loss(reconstructed, pixel_values)
        else:
            rec_loss = self.rec_loss(
                reconstructed,
                pixel_values,
                reduction='none',
            ).sum(dim=(1, 2, 3)).mean()

        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = rec_loss + self.kld_beta * kld_loss
        return loss, rec_loss, kld_loss

    def _simulation_z(self, batch_size: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.randn(batch_size, *self.hidden_shape, device=device)


class CVAEFilmMultipleFeatures(_BaseCVAEMultipleFeatures):
    """
    Pure-FiLM conditional variational autoencoder.

    Feature IDs are expected to be batched ``LongTensor[B]`` values, one per configured feature. Each feature is
    embedded into a spatial FiLM condition tensor and concatenated in deterministic ``feature_sizes`` order.
    ``conditional_simulation`` always returns batched image logits or images.
    """

    def __init__(
            self,
            config: AEModelConfig,
            feature_sizes: dict[str, int],
            n_cond_channels: int,
            image_size: int,
            reduction: Literal['mean', 'sum'] = 'mean',
    ) -> None:
        """
        Initialize the factor-conditioned FiLM CVAE.

        Args:
            config (AEModelConfig): Model dimension and VAE loss configuration.
            feature_sizes (dict[str, int]): Categorical cardinality for each conditioning feature.
            n_cond_channels (int): Number of FiLM condition channels per feature.
            image_size (int): Square input image size.
            reduction (Literal['mean', 'sum']): Reconstruction loss reduction mode.
        """
        super().__init__(
            config=config,
            feature_sizes=feature_sizes,
            n_cond_channels=n_cond_channels,
            image_size=image_size,
            reduction=reduction,
        )
        total_cond_channels = n_cond_channels * len(self.feature_names)
        decoder_config = replace(config, cond_channels=total_cond_channels)
        self.image_decoder = DecoderUpMFiLM(decoder_config)

    def build_condition(self, feature_ids: dict[str, torch.LongTensor]) -> torch.Tensor:
        """
        Build the decoder FiLM condition tensor.

        Args:
            feature_ids (dict[str, torch.LongTensor]): Batched categorical IDs by feature name.

        Returns:
            torch.Tensor: Decoder condition tensor.
        """
        return self.embed_features(feature_ids)

    def decode_from_z(
            self,
            z: torch.Tensor,
            condition: torch.Tensor,
            *,
            return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Decode a latent tensor using a FiLM condition.

        Args:
            z (torch.Tensor): Latent tensor sampled from the encoder or prior.
            condition (torch.Tensor): Concatenated FiLM condition tensor.
            return_features (bool): Whether to return decoder penultimate features.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Image logits, optionally with decoder features.
        """
        return self.image_decoder(z, condition, return_features=return_features)

    def forward(
            self,
            pixel_values: torch.Tensor,
            feature_ids: dict[str, torch.LongTensor],
    ) -> VAEOutput:
        """
        Run a training forward pass.

        Args:
            pixel_values (torch.Tensor): Batched input images.
            feature_ids (dict[str, torch.LongTensor]): Batched categorical IDs by feature name.

        Returns:
            VAEOutput: Reconstruction and VAE loss bundle.
        """
        mu, logvar = self.encode(pixel_values)
        z = self._reparameterize(mu, logvar)
        condition = self.build_condition(feature_ids)
        reconstructed = self.decode_from_z(z, condition)
        loss, rec_loss, kld_loss = self.vae_loss(
            reconstructed,
            pixel_values,
            mu,
            logvar,
        )
        return VAEOutput(
            rec=reconstructed,
            loss=loss,
            rec_loss=rec_loss,
            kld_loss=kld_loss,
        )

    @torch.no_grad()
    def conditional_simulation(
            self,
            *,
            feature_ids: dict[str, torch.LongTensor],
    ) -> torch.Tensor:
        """
        Sample images from the prior under the requested feature condition.

        Args:
            feature_ids (dict[str, torch.LongTensor]): Batched categorical IDs by feature name.
            return_logits (bool): Whether to return raw logits instead of sigmoid images.

        Returns:
            torch.Tensor: Batched generated image logits or image values.
        """
        batch_size = next(iter(feature_ids.values())).shape[0]
        z = self._simulation_z(batch_size)
        condition = self.build_condition(feature_ids)
        reconstructed = self.decode_from_z(z, condition)
        return reconstructed


class CVAEFilmMultipleFeaturesLimitedCapacity(CVAEFilmMultipleFeatures):
    """
    FiLM CVAE with limited-capacity Beta VAE objective.
    """

    def __init__(
        self,
        config: LimitedCapacityAEModelConfig,
        feature_sizes: dict[str, int],
        n_cond_channels: int,
        image_size: int,
        reduction: Literal['mean', 'sum'] = 'mean',
    ) -> None:
        """
        Initialize the limited-capacity factor-conditioned FiLM CVAE.

        Args:
            config (LimitedCapacityAEModelConfig): Model dimensions and capacity-loss configuration.
            feature_sizes (dict[str, int]): Categorical cardinality for each conditioning feature.
            n_cond_channels (int): Number of FiLM condition channels per feature.
            image_size (int): Square input image size.
            reduction (Literal['mean', 'sum']): Accepted for constructor compatibility.
        """
        super().__init__(
            config=config,
            feature_sizes=feature_sizes,
            n_cond_channels=n_cond_channels,
            image_size=image_size,
            reduction=reduction,
        )
        self.global_step = 0

    def _capacity_schedule(self) -> torch.Tensor:
        if self.config.C_iters_to_max <= 0:
            return torch.tensor(float(self.config.C_max))

        t = min(self.global_step / self.config.C_iters_to_max, 1.0)
        capacity = self.config.C_start + t * (self.config.C_max - self.config.C_start)
        return torch.tensor(float(capacity))

    def vae_loss(
            self,
            reconstructed: torch.Tensor,
            pixel_values: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the limited-capacity CVAE reconstruction and KL losses.

        Args:
            reconstructed (torch.Tensor): Decoder image logits.
            pixel_values (torch.Tensor): Target image values.
            mu (torch.Tensor): Latent mean tensor.
            logvar (torch.Tensor): Latent log-variance tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Total, reconstruction, and KL losses.
        """
        capacity = self._capacity_schedule().to(device=mu.device)
        if self.training:
            self.global_step += 1

        rec_loss = self.rec_loss(
            reconstructed,
            pixel_values,
            reduction='none',
        ).sum(dim=(1, 2, 3))
        kld_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(1, 2, 3))
        loss = rec_loss + self.config.gamma * torch.abs(kld_loss - capacity)
        return loss.mean(), rec_loss.mean(), kld_loss.mean()
