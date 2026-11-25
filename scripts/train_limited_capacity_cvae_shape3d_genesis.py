import sys
import warnings

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torchvision.utils as vutils
from typing import Any
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd()))
warnings.simplefilter('ignore')

from data import get_Shapes3D_dataloaders, get_Shapes3D_datasets
from configs import shape3d_feature_config_genesis
from models import (
    LimitedCapacityAEModelConfig, CVAEFilmMultipleFeaturesLimitedCapacity
)
from utils import filepaths, constants
from training import CVAETrainer, TrainerConfig

device = torch.device("cuda")

###

def build_model(
    *,
    latent_dim: int,
    hidden_dim_base: int,
    kld_beta: float,
    n_cond_channels: int,
    C_max: int,
    gamma: int,
    C_iters_to_max: int,
    feature_config: dict[str, Any],
) -> Any:
    
    config = LimitedCapacityAEModelConfig(
        latent_dim=latent_dim,
        hidden_dim_base=hidden_dim_base,
        kld_beta=kld_beta,
        C_max=C_max,
        gamma=gamma,
        C_iters_to_max=C_iters_to_max
    )

    return CVAEFilmMultipleFeaturesLimitedCapacity(
        config=config,
        feature_sizes={key: len(feature_config[key]) for key in feature_config},
        n_cond_channels=n_cond_channels,
        image_size=constants.SHAPE3D_IMAGE_SIZE,
    )
    
###

feature_config = shape3d_feature_config_genesis.feature_config

MODEL_PARS_PATH = (
    filepaths.EXPERIMENTS_DIR / 
    'cvae_shape3d_model_selection' / 
    'best_model_params.npy'
)

model_pars = np.load(MODEL_PARS_PATH, allow_pickle=True).item()

### Dataloaders

train_ds, test_ds, _ = get_Shapes3D_datasets(
    file_path=filepaths.SHAPES3D_FILE_PATH,
    **feature_config,
)

train_loader, test_loader = get_Shapes3D_dataloaders(
    train_ds,
    test_ds,
    batch_size=32,
    num_workers=0,
)

### Training

for C in [0, 40, 80, 200, 400]:

    model = build_model(
        **model_pars,
        C_max=C,
        gamma=30,
        C_iters_to_max=10000,
        feature_config=feature_config
    )

    trainer = CVAETrainer(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        active_features=list(feature_config.keys()),
        save_path=filepaths.EXPERIMENTS_DIR / 'limited_capacity' / f"capacity_{C}",
        experiment_id="",
        trainer_config=TrainerConfig(
            lr=0.0015,
            max_epochs=20,
            device=device,
            show_progress=True,
            save_checkpoints=False,
        ),
    )

    history = trainer.fit()