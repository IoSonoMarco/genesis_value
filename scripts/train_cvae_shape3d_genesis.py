import sys
import warnings

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torchvision.utils as vutils
import yaml
from typing import Any
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd()))
warnings.simplefilter('ignore')

from data import get_Shapes3D_dataloaders, get_Shapes3D_datasets
from configs import shape3d_feature_config_genesis
from models import AEModelConfig, CVAEFilmMultipleFeatures
from utils import filepaths, constants
from training import CVAETrainer, TrainerConfig

###

MODEL_PARS_PATH = (
    filepaths.EXPERIMENTS_DIR / 
    'cvae_shape3d_model_selection' / 
    'best_model_params.npy'
)
DEFAULT_EXPERIMENT_ID = 'shape3d_genesis'
DEFAULT_TRAINING_ARGS = {
    'experiment_id': None,
    'epochs': 30,
    'batch_size': 16,
    'lr': 0.0015,
    'test_only': False,
    'model_pars': None,
}

###

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=Path, default=None)
    config_args, _ = config_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description='Train CVAE Shapes3D.')
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--experiment_id', type=str, default=DEFAULT_TRAINING_ARGS['experiment_id'])
    parser.add_argument('--epochs', type=int, default=DEFAULT_TRAINING_ARGS['epochs'])
    parser.add_argument('--batch-size', type=int, default=DEFAULT_TRAINING_ARGS['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULT_TRAINING_ARGS['lr'])
    parser.add_argument('--test-only', action='store_true', default=DEFAULT_TRAINING_ARGS['test_only'])
    parser.set_defaults(model_pars=DEFAULT_TRAINING_ARGS['model_pars'])

    if config_args.config is not None:
        config = yaml.safe_load(config_args.config.read_text(encoding='utf-8'))
        config_defaults = {**DEFAULT_TRAINING_ARGS, **config}
        parser.set_defaults(**config_defaults)
        return parser.parse_args(['--config', str(config_args.config)])

    return parser.parse_args(argv)

###

def resolve_model_pars(args: argparse.Namespace) -> dict[str, Any]:
    if args.model_pars is not None:
        return args.model_pars

    return np.load(MODEL_PARS_PATH, allow_pickle=True).item()


def build_model(
    *,
    latent_dim: int,
    hidden_dim_base: int,
    kld_beta: float,
    n_cond_channels: int,
    feature_config: dict[str, Any],
) -> Any:
    config = AEModelConfig(
        latent_dim=latent_dim,
        hidden_dim_base=hidden_dim_base,
        m=3,
        kld_beta=kld_beta,
    )

    return CVAEFilmMultipleFeatures(
        config=config,
        feature_sizes={key: len(feature_config[key]) for key in feature_config},
        n_cond_channels=n_cond_channels,
        image_size=constants.SHAPE3D_IMAGE_SIZE,
    )
    
def model_testing(model, test_loader, img_name, experiment_id):
    with torch.no_grad():
        model.eval()
        device = next(iter(model.parameters())).device
        
        batch = next(iter(test_loader))
        batch.to(device)
        
        out = model(
            pixel_values=batch.pixel_values,
            feature_ids=batch.factors.todict()
        )
        rec = torch.sigmoid(out.rec).cpu()
        
        rec_grid = vutils.make_grid(rec, nrow=4, padding=2)
        true_grid = vutils.make_grid(batch.pixel_values.cpu(), nrow=4, padding=2)
        
    fig, axes = plt.subplots(1,2,figsize=(8, 8))

    axes[0].imshow(true_grid.permute(1, 2, 0).numpy())
    axes[0].set_title("True", fontsize=16)
    axes[1].imshow(rec_grid.permute(1, 2, 0).numpy())
    axes[1].set_title("Rec", fontsize=16)

    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout(pad=1)
    
    savepath = (
        filepaths.VISUALS_DIR / 
        experiment_id / 
        "image_recon"
    )
    savepath.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(savepath / f"{img_name}.pdf",bbox_inches="tight")
    
def main() -> None:
    args = parse_args()
    
    experiment_id = (
        args.experiment_id if 
        args.experiment_id is not None 
        else DEFAULT_EXPERIMENT_ID
    )
    SAVE_PATH = filepaths.OUTPUT_DIR / "cvae" / experiment_id
    
    print("CVAE Shape3D Training")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("... \n")
    
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    feature_config = shape3d_feature_config_genesis.feature_config
    
    print("Load Model from best Model Selection Parameters... \n")
    model_pars = resolve_model_pars(args)
    print(model_pars)
    model = build_model(
        **model_pars,
        feature_config=feature_config
    )
    
    print("Load Dataset... \n")
    
    train_ds, test_ds, _ = get_Shapes3D_datasets(
        file_path=filepaths.SHAPES3D_FILE_PATH,
        **feature_config,
    )

    train_loader, test_loader = get_Shapes3D_dataloaders(
        train_ds,
        test_ds,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=device.type == 'cuda',
    )
    
    if args.test_only:
        w_path = SAVE_PATH / "ckpt" / "best.pt"
        if not w_path.exists():
            raise ValueError("The model has not benn trained yet!")
        
        print("Load weights... \n")
        w = torch.load(w_path, weights_only=False, map_location="cpu")
        model.load_state_dict(w)
        model.to(device)
        
        model_testing(model, train_loader, "b1", experiment_id)
        
        return
            
    
    trainer = CVAETrainer(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        active_features=list(feature_config.keys()),
        save_path=SAVE_PATH,
        experiment_id="ckpt",
        trainer_config=TrainerConfig(
            lr=args.lr,
            max_epochs=args.epochs,
            device=device,
            show_progress=True,
            save_checkpoints=True,
        ),
    )
    
    trainer.fit()
    
    np.save(SAVE_PATH / "model_params", model_pars)
    
    
if __name__ == '__main__':
    main()
