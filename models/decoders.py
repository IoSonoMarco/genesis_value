import torch
import torch.nn as nn
from models.utils import SkipConnectionWrapper, FiLM2d
from models.base import AEModelConfig, BaseModel


class DecoderUpM(BaseModel):
    """Convolutional image decoder for spatial latent maps.

    ``DecoderUpM`` upsamples a latent tensor of shape
    ``(batch, latent_dim, h, w)`` by a factor of ``2 ** m`` in each spatial
    dimension and returns RGB image logits. For example, with ``m=3`` and a
    latent spatial size of ``4 x 4``, the output spatial size is ``32 x 32``.

    The decoder first projects ``latent_dim`` channels to
    ``hidden_dim_base * 2**m`` channels, then applies ``m`` transposed-conv
    upsampling blocks with residual convolutional refinement. The final
    ``output_proj`` maps the last hidden map to 3 RGB logit channels.

    Args:
        config: Architecture configuration. Uses ``latent_dim``,
            ``hidden_dim_base``, and ``m``.
    """
    def __init__(self, config: AEModelConfig):
        """
        Config has the following
        latent_dim: number of channels of the compressed image. It provides the base input to the decoder.
        hidden_dim_base: the seed number of convolutional filters used in the Encoder.
        m: downsampling steps.
        """
        super().__init__(config=config)
        self.latent_dim = config.latent_dim
        self.hidden_dim_base = config.hidden_dim_base
        self.m = config.m

        Cout = self.hidden_dim_base * (2**self.m)
        self.inverse_hidden_proj = nn.Sequential(
            nn.Conv2d(self.latent_dim, Cout, 1),
            nn.BatchNorm2d(Cout),
            nn.ReLU()
        )

        layers = []
        for i in range(self.m, 0, -1):
            Cin = self.hidden_dim_base * (2 ** i)
            Cout = self.hidden_dim_base * (2 ** (i - 1))

            upsample_block = nn.ConvTranspose2d(
                Cin, Cout,
                kernel_size=4,
                stride=2, padding=1,
                output_padding=0
            )

            residual_block = nn.Sequential(
                nn.Conv2d(
                    Cout, Cout,
                    kernel_size=3,
                    stride=1, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    Cout, Cout,
                    kernel_size=3,
                    stride=1, padding=1
                )
            )
            residual_block = SkipConnectionWrapper(residual_block, Cout, Cout)

            layers.append(
                nn.Sequential(
                    upsample_block,
                    nn.ReLU(),
                    residual_block,
                    nn.BatchNorm2d(Cout)
                )
            )

        self.layers = nn.Sequential(*layers)

        self.output_proj = nn.Conv2d(Cout, 3, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.inverse_hidden_proj(z)
        x = self.layers(x)
        x = self.output_proj(x)
        return x
    

class DecoderUpMFiLM(BaseModel):
    """FiLM-conditioned convolutional decoder for pure-FiLM CVAE models.

    This decoder has the same upsampling topology as ``DecoderUpM`` but applies
    spatial FiLM modulation after batch normalization in each upsampling stage.
    ``z`` is the image decoder latent and ``cond`` is the concatenated FiLM
    condition tensor. In the pure-FiLM CVAE architecture, stimulus and reward
    condition embeddings should be passed only through ``cond``; they should
    not be concatenated into ``z`` before calling this decoder.

    Args:
        config: Architecture configuration. Uses ``latent_dim``,
            ``hidden_dim_base``, ``m``, and ``cond_channels``. ``cond_channels``
            must match ``cond.shape[1]`` at runtime.

    Forward Args:
        z: Spatial latent tensor with shape
            ``(batch, latent_dim, hidden_h, hidden_w)``.
        cond: Spatial FiLM condition tensor with shape
            ``(batch, cond_channels, cond_h, cond_w)``. Each FiLM block
            interpolates this tensor to its current hidden resolution.
        return_features: If ``False``, return only RGB logits. If ``True``,
            return ``(logits, penultimate_features)`` where
            ``penultimate_features`` is the final hidden representation before
            ``output_proj`` and has ``hidden_dim_base`` channels.
    """
    def __init__(self, config: AEModelConfig):
        super().__init__(config=config)
        self.latent_dim = config.latent_dim
        self.hidden_dim_base = config.hidden_dim_base
        self.m = config.m
        self.cond_channels = config.cond_channels

        Cout = self.hidden_dim_base * (2**self.m)
        self.inverse_hidden_proj = nn.Sequential(
            nn.Conv2d(self.latent_dim, Cout, 1),
            nn.BatchNorm2d(Cout),
            nn.ReLU()
        )

        blocks = []
        for i in range(self.m, 0, -1):
            Cin = self.hidden_dim_base * (2 ** i)
            Cout = self.hidden_dim_base * (2 ** (i - 1))

            upsample_block = nn.ConvTranspose2d(Cin, Cout, kernel_size=4, stride=2, padding=1)

            residual_block = nn.Sequential(
                nn.Conv2d(Cout, Cout, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(Cout, Cout, kernel_size=3, stride=1, padding=1)
            )
            residual_block = SkipConnectionWrapper(residual_block, Cout, Cout)

            blocks.append(nn.ModuleDict({
                "up": upsample_block,
                "act": nn.ReLU(),
                "residual": residual_block,
                "bn": nn.BatchNorm2d(Cout),
                "film": FiLM2d(self.cond_channels, Cout)
            }))

        self.blocks = nn.ModuleList(blocks)
        self.output_proj = nn.Conv2d(Cout, 3, kernel_size=3, padding=1)

    def forward(
            self,
            z: torch.Tensor,
            cond: torch.Tensor,
            return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.inverse_hidden_proj(z)

        for blk in self.blocks:
            x = blk["up"](x)
            x = blk["act"](x)
            x = blk["residual"](x)
            x = blk["bn"](x)
            x = blk["film"](x, cond)

        penultimate_features = x
        x = self.output_proj(x)

        if return_features:
            return x, penultimate_features

        return x
