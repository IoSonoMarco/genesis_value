import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter('ignore')

from configs import shape3d_feature_config_debugging
from data import get_Shapes3D_datasets
from data import shapes3d_feature_reward_fn
from models import LinearDecoder
from utils import filepaths


OUTPUT_DIR = filepaths.ROOT / 'output' / 'reward_linear_decoder' / 'shape3d_feature_debugging'
IGT_OUTPUT_DIR = filepaths.OUTPUT_DIR / 'reward_linear_decoder' / "igt"


def make_reward_scope(rewards: np.ndarray) -> np.ndarray:
    """
    Build a dense scalar reward range for decoder training.

    Args:
        rewards (np.ndarray): Reward values observed in the Shapes3D dataset.

    Returns:
        np.ndarray: Reward scope with shape ``[N]``.
    """
    return np.arange(-(rewards[-1] + 2), rewards[-1] + 2, 0.05)


def make_igt_reward_scope() -> np.ndarray:
    return np.linspace(-15.0, 15.0, 601)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a scalar reward linear decoder.')
    parser.add_argument('--igt', action='store_true')
    return parser.parse_args(argv)


def get_training_scope(args: argparse.Namespace) -> np.ndarray:
    if args.igt:
        return make_igt_reward_scope()

    *_, reward_values = get_Shapes3D_datasets(
        file_path=filepaths.SHAPES3D_FILE_PATH,
        **shape3d_feature_config_debugging.feature_config,
        reward_fn=shapes3d_feature_reward_fn,
    )
    return make_reward_scope(reward_values)


def resolve_weights_path(args: argparse.Namespace) -> Path:
    if args.igt:
        return IGT_OUTPUT_DIR / 'weights.pt'

    return OUTPUT_DIR / 'weights.pt'


model_config = dict(
    hidden_dim=4,
    latent_dim=8,
)


def main() -> None:
    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    scope = get_training_scope(args)
    weights_path = resolve_weights_path(args)

    model = LinearDecoder(**model_config)
    model.to(device)
    model.set_dataloader(scope, batch_size=8)
    model.train_model(epochs=50, lr=0.01)

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        state_dict=model.state_dict(),
        model_config=model_config
    ), weights_path)
    print(f'Saved reward linear decoder weights to {weights_path}')


if __name__ == '__main__':
    main()
