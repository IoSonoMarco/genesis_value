"""
Load YAML definitions for GENESIS reward-bandit simulations.
"""

from pathlib import Path
from typing import Any

import yaml


def load_simulation_config(config_path: Path) -> dict[str, Any]:
    """
    Load the complete simulation definition from YAML.

    Args:
        config_path (Path): Simulation YAML configuration path.

    Returns:
        dict[str, Any]: Experiment name, sample counts, and simulation grid.
    """
    return yaml.safe_load(config_path.read_text(encoding='utf-8'))


def get_replacement_output_dir(
    *,
    root_dir: Path,
    output_name: str,
    sample_replacement: bool,
) -> Path:
    """
    Resolve a condition-specific output directory.

    Args:
        root_dir (Path): Parent directory for experiment or analysis outputs.
        output_name (str): Experiment or analysis output directory name.
        sample_replacement (bool): Whether episodic samples may repeat.

    Returns:
        Path: Directory dedicated to the replacement condition.
    """
    replacement_name = str(sample_replacement).lower()

    return root_dir / output_name / f'sample_replacement_{replacement_name}'
