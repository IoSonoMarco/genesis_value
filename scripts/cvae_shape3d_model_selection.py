from __future__ import annotations

import argparse
import json
import sys
import pandas as pd
import numpy as np
import warnings
from itertools import product
from pathlib import Path
from typing import Any

import yaml

import torch

sys.path.insert(0, str(Path.cwd()))
warnings.simplefilter('ignore')

from configs import shape3d_feature_config_model_selection
from data import get_Shapes3D_dataloaders, get_Shapes3D_datasets
from training import CVAETrainer, TrainerConfig
from utils import filepaths, constants
from models import AEModelConfig, CVAEFilmMultipleFeatures


DEFAULT_MODEL_CONFIG_PATH = filepaths.ROOT / 'configs' / 'cvae_shape3d_model_selection.yaml'
DEFAULT_EXPERIMENT_ID = 'cvae_shape3d_model_selection'
MODEL_PARAMETER_TYPES = {
    'latent_dim': int,
    'hidden_dim_base': int,
    'kld_beta': float,
    'n_cond_channels': int,
}


def build_cell_id(*, model_pars: dict[str, Any], grid_keys: list[str]) -> str:
    parts = [f'{key}={model_pars[key]}' for key in grid_keys]
    return '__'.join(parts)


def expand_model_configs(
    *,
    selection_config: dict[str, Any],
    fixed_model_pars: dict[str, Any],
) -> list[dict[str, Any]]:
    grid = selection_config['grid']
    grid_keys = list(grid.keys())
    cells = []

    unknown_keys = sorted(set(fixed_model_pars) - set(grid))
    if unknown_keys:
        raise ValueError(f'Unknown fixed model parameter(s): {unknown_keys}')

    for key, value in fixed_model_pars.items():
        if value not in grid[key]:
            raise ValueError(
                f'Fixed value {key}={value} is not in the model grid. Allowed values: {grid[key]}',
            )

    sweep_values = [
        [fixed_model_pars[key]] if key in fixed_model_pars else grid[key]
        for key in grid_keys
    ]
    fixed_grid_keys = [key for key in grid_keys if key in fixed_model_pars]
    selection_id = (
        build_cell_id(model_pars=fixed_model_pars, grid_keys=fixed_grid_keys)
        if fixed_grid_keys
        else 'full_grid'
    )

    for values in product(*sweep_values):
        model_pars = dict(zip(grid_keys, values))
        cells.append({
            'selection_id': selection_id,
            'cell_id': build_cell_id(model_pars=model_pars, grid_keys=grid_keys),
            'model_pars': model_pars,
        })

    return cells


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run CVAE Shapes3D model-selection cells.')
    parser.add_argument('--model-config-path', type=Path, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument('--latent_dim', '--latent-dim', dest='latent_dim', type=int, default=None)
    parser.add_argument('--hidden_dim_base', '--hidden-dim-base', dest='hidden_dim_base', type=int, default=None)
    parser.add_argument('--kld_beta', '--kld-beta', dest='kld_beta', type=float, default=None)
    parser.add_argument('--n_cond_channels', '--n-cond-channels', dest='n_cond_channels', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--experiment-id', type=str, default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument('--save-path', type=Path, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    return parser.parse_args(argv)


def fixed_model_pars_from_args(args: argparse.Namespace) -> dict[str, Any]:
    fixed_model_pars = {
        key: getattr(args, key)
        for key in MODEL_PARAMETER_TYPES
        if getattr(args, key) is not None
    }
    return fixed_model_pars


def parse_cell_model_pars(cell_id: str) -> dict[str, Any]:
    model_pars = {}
    for part in cell_id.split('__'):
        key, value = part.split('=')
        model_pars[key] = MODEL_PARAMETER_TYPES[key](value)

    return model_pars


def build_history_from_summary(summary: dict[str, Any]) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {
        'train_loss': [],
        'train_rec': [],
        'train_kld': [],
        'test_loss': [],
        'test_rec': [],
        'test_kld': [],
    }

    for epoch in summary['epochs']:
        for split in ['train', 'test']:
            for metric in ['loss', 'rec', 'kld']:
                history[f'{split}_{metric}'].append(epoch[split][metric])

    return history


def build_result_from_summary(
    *,
    experiment_id: str,
    cell_id: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    best_epoch = summary['best_epoch']
    best_record = summary['epochs'][best_epoch - 1]
    history = build_history_from_summary(summary)

    return {
        'experiment_id': experiment_id,
        'selection_id': 'posthoc_summary',
        'cell_id': cell_id,
        **parse_cell_model_pars(cell_id),
        'max_epochs': summary['max_epochs'],
        'best_epoch': best_epoch,
        'best_test_rec': summary['best_test_rec'],
        'best_test_loss': best_record['test']['loss'],
        'best_test_kld': best_record['test']['kld'],
        'history': history,
    }


def summarize_model_selection_results(
    *,
    experiment_dir: Path,
    experiment_id: str,
) -> Path:
    results_path = experiment_dir / 'results.jsonl'
    summary_paths = sorted((experiment_dir / 'summaries').glob('*/summary.json'))
    results = []

    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        result = build_result_from_summary(
            experiment_id=experiment_id,
            cell_id=summary_path.parent.name,
            summary=summary,
        )
        results.append(result)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        ''.join(f'{json.dumps(result)}\n' for result in results),
        encoding='utf-8',
    )

    return results_path


def run_model_selection(args: argparse.Namespace) -> Path:
    # Load the YAML grid before training setup so invalid slices fail fast.
    selection_config = yaml.safe_load(Path(args.model_config_path).read_text(encoding='utf-8'))
    cells = expand_model_configs(
        selection_config=selection_config,
        fixed_model_pars=fixed_model_pars_from_args(args),
    )

    feature_config = shape3d_feature_config_model_selection.feature_config
    save_path = args.save_path or filepaths.EXPERIMENTS_DIR
    experiment_dir = save_path / args.experiment_id
    device = torch.device(args.device) if args.device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu',
    )

    train_ds, test_ds, _ = get_Shapes3D_datasets(
        file_path=filepaths.SHAPES3D_FILE_PATH,
        **feature_config,
    )
    train_loader, test_loader = get_Shapes3D_dataloaders(
        train_ds,
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    print(f'Running {len(cells)} model-selection cell(s) on {device}')
    for index, cell in enumerate(cells, start=1):
        print(f'Cell {index}/{len(cells)} | {cell["selection_id"]} | {cell["cell_id"]}')
        model = build_model(
            **cell['model_pars'],
            feature_config=feature_config,
        )
        trainer = CVAETrainer(
            model=model,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            active_features=list(feature_config.keys()),
            save_path=experiment_dir / 'summaries',
            experiment_id=cell['cell_id'],
            trainer_config=TrainerConfig(
                lr=args.lr,
                max_epochs=args.epochs,
                device=device,
                show_progress=not args.no_progress,
                save_checkpoints=False,
            ),
        )
        history = trainer.fit()
        print(f'Finished {cell["cell_id"]} with {len(history["test_rec"])} epoch(s)')

    return experiment_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    save_path = args.save_path or filepaths.EXPERIMENTS_DIR
    experiment_dir = save_path / args.experiment_id

    if args.summarize_only:
        results_path = summarize_model_selection_results(
            experiment_dir=experiment_dir,
            experiment_id=args.experiment_id,
        )
        print(f'Wrote model-selection summary to {results_path}')
        return
    
    
    if args.analyze_only:
        print("Analyzing model selection experiment data...")
        file_path = experiment_dir / 'results.jsonl'
        if not file_path.exists():
            raise ValueError("results.jsonl has not been created yet...")
        
        df = pd.read_json(path_or_buf=file_path, lines=True)
        out = (
            df.sort_values("best_test_rec")
            .iloc[0][[
                "latent_dim", 
                "hidden_dim_base", 
                "kld_beta", 
                "n_cond_channels"
            ]]
        )
        latent_dim, hidden_dim_base, kld_beta, n_cond_channels = out
        
        model_params = dict(
            latent_dim=latent_dim,
            hidden_dim_base=hidden_dim_base,
            kld_beta=kld_beta,
            n_cond_channels=n_cond_channels
        )
        np.save(experiment_dir / 'best_model_params', model_params)
        print(f'Best model parameters saved to {experiment_dir}')
        return
        

    run_model_selection(args)


if __name__ == '__main__':
    main()
