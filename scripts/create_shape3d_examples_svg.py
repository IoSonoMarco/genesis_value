import base64
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree

import sys
import warnings
sys.path.insert(0, str(Path.cwd()))
warnings.simplefilter('ignore')

import h5py
import matplotlib.image
import numpy as np
import pandas as pd

from utils import filepaths


CLASS_COLUMNS = [
    'object_shape',
    'object_color',
    'floor_color',
    'wall_color',
]
FACTOR_MAP = {
    'floor_color': 0,
    'wall_color': 1,
    'object_color': 2,
    'object_scale': 3,
    'object_shape': 4,
    'object_orientation': 5,
}
FEATURE_CONFIG = {
    'object_shape': [0, 1, 2, 3],
    'object_color': [0, 2, 3, 5, 7],
    'floor_color': [0, 2, 3, 5, 7],
    'wall_color': [0, 2, 3, 5, 7],
}
SVG_NAMESPACE = 'http://www.w3.org/2000/svg'
SVG_WIDTH = 2240
SVG_HEIGHT = 720
TILE_SIZE = 270
TILE_X_OFFSET = 300
TILE_Y_OFFSET = 310
TILE_X_START = 80
TILE_Y_START = 82
CUBE_SVG_WIDTH = 1640
CUBE_GRID_COLUMNS = 5
CUBE_CORNER_RADIUS = 24
CUBE_BORDER_WIDTH = 4


def load_shape3d_data_table(*, file_path: str, feature_config: dict[str, list[int]]) -> pd.DataFrame:
    """
    Load configured Shape3D metadata with scale and orientation variants.

    Args:
        file_path (str): Path to the Shape3D HDF5 dataset.
        feature_config (dict[str, list[int]]): Global semantic factor values to retain.

    Returns:
        pd.DataFrame: One row per configured Shape3D image.
    """
    with h5py.File(file_path, 'r') as dataset_file:
        raw_labels = dataset_file['labels'][:]

    data_table = pd.DataFrame({'h5_idx': np.arange(len(raw_labels))})

    for factor_name, factor_idx in FACTOR_MAP.items():
        raw_values = raw_labels[:, factor_idx]
        unique_values = np.unique(raw_values)
        value_to_idx = {value: idx for idx, value in enumerate(unique_values)}
        data_table[factor_name] = np.array([value_to_idx[value] for value in raw_values])

    mask = np.ones(len(data_table), dtype=bool)

    for factor_name in CLASS_COLUMNS:
        mask &= data_table[factor_name].isin(feature_config[factor_name]).to_numpy()

    return data_table.loc[mask].reset_index(drop=True)


def sample_shape3d_image_families(*, data_table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Sample ten semantic classes and two three-image reference families.

    Args:
        data_table (pd.DataFrame): Shape3D metadata including image indices and factor columns.
        rng (np.random.Generator): Random sampler.

    Returns:
        pd.DataFrame: Fourteen selected images, with base samples first.
    """
    grouped = list(data_table.groupby(CLASS_COLUMNS, sort=False))
    eligible_groups = [group for _, group in grouped if len(group) >= 3]
    cube_group_indices = [
        group_idx
        for group_idx, group in enumerate(eligible_groups)
        if int(group['object_shape'].iloc[0]) == 0
    ]
    cube_group_idx = int(rng.choice(cube_group_indices))
    remaining_group_indices = [
        group_idx
        for group_idx in range(len(eligible_groups))
        if group_idx != cube_group_idx
    ]
    selected_group_indices = [
        cube_group_idx,
        *rng.choice(remaining_group_indices, size=9, replace=False).tolist(),
    ]
    base_samples = []

    for group_idx in selected_group_indices:
        group = eligible_groups[int(group_idx)]
        sample_idx = int(rng.integers(len(group)))
        base_samples.append(group.iloc[sample_idx])

    base_samples = pd.DataFrame(base_samples).reset_index(drop=True)
    cube_reference_idx = int(rng.choice(base_samples.index[base_samples['object_shape'] == 0]))
    other_reference_indices = [
        sample_idx
        for sample_idx in range(len(base_samples))
        if sample_idx != cube_reference_idx
    ]
    reference_indices = [
        cube_reference_idx,
        int(rng.choice(other_reference_indices)),
    ]
    companion_samples = []

    for reference_idx in reference_indices:
        reference = base_samples.iloc[int(reference_idx)]
        same_class = data_table.loc[
            (data_table[CLASS_COLUMNS] == reference[CLASS_COLUMNS]).all(axis=1)
            & (data_table['h5_idx'] != reference['h5_idx'])
            & (
                (data_table['object_scale'] != reference['object_scale'])
                | (data_table['object_orientation'] != reference['object_orientation'])
            )
        ]
        companion_indices = rng.choice(len(same_class), size=2, replace=False)
        companion_samples.extend(same_class.iloc[companion_indices].to_dict('records'))

    return pd.concat(
        [base_samples, pd.DataFrame(companion_samples)],
        ignore_index=True,
    )


def sample_cube_shape3d_images(*, data_table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Sample two scale variants for each configured cube color.

    Args:
        data_table (pd.DataFrame): Shape3D metadata including image indices and factor columns.
        rng (np.random.Generator): Random sampler.

    Returns:
        pd.DataFrame: Ten selected cube images.
    """
    cube_data = data_table.loc[data_table['object_shape'] == 0]
    samples = []

    for object_color in FEATURE_CONFIG['object_color']:
        color_data = cube_data.loc[cube_data['object_color'] == object_color]
        selected_scales = rng.choice(color_data['object_scale'].unique(), size=2, replace=False)

        for object_scale in selected_scales:
            scale_data = color_data.loc[color_data['object_scale'] == object_scale]
            sample_idx = int(rng.integers(len(scale_data)))
            samples.append(scale_data.iloc[sample_idx])

    return pd.DataFrame(samples).reset_index(drop=True)


def image_to_data_url(*, image: np.ndarray) -> str:
    """
    Encode one RGB image as an embedded PNG data URL.

    Args:
        image (np.ndarray): Shape3D RGB image.

    Returns:
        str: Base64 PNG data URL.
    """
    buffer = BytesIO()
    matplotlib.image.imsave(buffer, image, format='png')
    encoded_image = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded_image}'


def write_shape3d_examples_svg(*, file_path: str, output_path: Path, rng: np.random.Generator) -> None:
    """
    Write a two-by-seven Shape3D contact sheet with embedded, selectable tiles.

    Args:
        file_path (str): Path to the Shape3D HDF5 dataset.
        output_path (Path): SVG output path.
        rng (np.random.Generator): Random sampler.
    """
    data_table = load_shape3d_data_table(
        file_path=file_path,
        feature_config=FEATURE_CONFIG,
    )
    samples = sample_shape3d_image_families(data_table=data_table, rng=rng)
    ElementTree.register_namespace('', SVG_NAMESPACE)
    svg = ElementTree.Element(
        f'{{{SVG_NAMESPACE}}}svg',
        {
            'width': str(SVG_WIDTH),
            'height': str(SVG_HEIGHT),
            'viewBox': f'0 0 {SVG_WIDTH} {SVG_HEIGHT}',
            'role': 'img',
        },
    )
    ElementTree.SubElement(svg, f'{{{SVG_NAMESPACE}}}rect', {'width': '100%', 'height': '100%', 'fill': '#FFFFFF'})

    with h5py.File(file_path, 'r') as dataset_file:
        images = dataset_file['images']

        for sample_idx, sample in samples.iterrows():
            row_idx, column_idx = divmod(sample_idx, 7)
            group = ElementTree.SubElement(
                svg,
                f'{{{SVG_NAMESPACE}}}g',
                {'id': f'shape3d_image_{sample_idx + 1}'},
            )
            ElementTree.SubElement(
                group,
                f'{{{SVG_NAMESPACE}}}image',
                {
                    'id': f'shape3d_raster_{sample_idx + 1}',
                    'x': str(TILE_X_START + column_idx * TILE_X_OFFSET),
                    'y': str(TILE_Y_START + row_idx * TILE_Y_OFFSET),
                    'width': str(TILE_SIZE),
                    'height': str(TILE_SIZE),
                    'href': image_to_data_url(image=images[int(sample.h5_idx)]),
                },
            )

    ElementTree.ElementTree(svg).write(output_path, encoding='utf-8', xml_declaration=True)


def write_cube_shape3d_examples_svg(*, file_path: str, output_path: Path, rng: np.random.Generator) -> None:
    """
    Write a two-by-five cube contact sheet with rounded, bordered selectable tiles.

    Args:
        file_path (str): Path to the Shape3D HDF5 dataset.
        output_path (Path): SVG output path.
        rng (np.random.Generator): Random sampler.
    """
    data_table = load_shape3d_data_table(
        file_path=file_path,
        feature_config=FEATURE_CONFIG,
    )
    samples = sample_cube_shape3d_images(data_table=data_table, rng=rng)
    ElementTree.register_namespace('', SVG_NAMESPACE)
    svg = ElementTree.Element(
        f'{{{SVG_NAMESPACE}}}svg',
        {
            'width': str(CUBE_SVG_WIDTH),
            'height': str(SVG_HEIGHT),
            'viewBox': f'0 0 {CUBE_SVG_WIDTH} {SVG_HEIGHT}',
            'role': 'img',
        },
    )
    ElementTree.SubElement(svg, f'{{{SVG_NAMESPACE}}}rect', {'width': '100%', 'height': '100%', 'fill': '#FFFFFF'})
    definitions = ElementTree.SubElement(svg, f'{{{SVG_NAMESPACE}}}defs')

    with h5py.File(file_path, 'r') as dataset_file:
        images = dataset_file['images']

        for sample_idx, sample in samples.iterrows():
            row_idx, column_idx = divmod(sample_idx, CUBE_GRID_COLUMNS)
            x_position = TILE_X_START + column_idx * TILE_X_OFFSET
            y_position = TILE_Y_START + row_idx * TILE_Y_OFFSET
            clip_path_id = f'shape3d_cube_clip_{sample_idx + 1}'
            clip_path = ElementTree.SubElement(
                definitions,
                f'{{{SVG_NAMESPACE}}}clipPath',
                {'id': clip_path_id},
            )
            ElementTree.SubElement(
                clip_path,
                f'{{{SVG_NAMESPACE}}}rect',
                {
                    'x': str(x_position),
                    'y': str(y_position),
                    'width': str(TILE_SIZE),
                    'height': str(TILE_SIZE),
                    'rx': str(CUBE_CORNER_RADIUS),
                },
            )
            group = ElementTree.SubElement(
                svg,
                f'{{{SVG_NAMESPACE}}}g',
                {'id': f'shape3d_cube_image_{sample_idx + 1}'},
            )
            ElementTree.SubElement(
                group,
                f'{{{SVG_NAMESPACE}}}image',
                {
                    'id': f'shape3d_cube_raster_{sample_idx + 1}',
                    'x': str(x_position),
                    'y': str(y_position),
                    'width': str(TILE_SIZE),
                    'height': str(TILE_SIZE),
                    'clip-path': f'url(#{clip_path_id})',
                    'href': image_to_data_url(image=images[int(sample.h5_idx)]),
                },
            )
            ElementTree.SubElement(
                group,
                f'{{{SVG_NAMESPACE}}}rect',
                {
                    'id': f'shape3d_cube_border_{sample_idx + 1}',
                    'x': str(x_position),
                    'y': str(y_position),
                    'width': str(TILE_SIZE),
                    'height': str(TILE_SIZE),
                    'rx': str(CUBE_CORNER_RADIUS),
                    'fill': 'none',
                    'stroke': '#000000',
                    'stroke-width': str(CUBE_BORDER_WIDTH),
                },
            )

    ElementTree.ElementTree(svg).write(output_path, encoding='utf-8', xml_declaration=True)


def main() -> None:
    """
    Generate the cube Shape3D example contact sheet.
    """
    write_cube_shape3d_examples_svg(
        file_path=filepaths.SHAPES3D_FILE_PATH,
        output_path=filepaths.VISUALS_DIR / 'shape3d_cube_examples_2x5.svg',
        rng=np.random.default_rng(44),
    )


if __name__ == '__main__':
    main()
