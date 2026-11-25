from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Callable, Iterable, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


RewardFn = Callable[[pd.Series | dict], float]


# ---------------------------------------------------------------------
# Semantic enums
# ---------------------------------------------------------------------

class HueColor(IntEnum):
    RED = 0
    ORANGE = 1
    YELLOW_GREEN = 2
    GREEN = 3
    TURQUOISE = 4
    CYAN = 5
    BLUE = 6
    VIOLET = 7
    MAGENTA = 8
    PINK = 9


class Shape(IntEnum):
    CUBE = 0
    CYLINDER = 1
    SPHERE = 2
    PILL = 3


# ---------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------

def shapes3d_feature_reward_fn(
    row: pd.Series | dict,
    quantize: bool = False,
) -> float:
    """
    Reward based only on:
        - shape
        - obj_hue
        - floor_hue
        - wall_hue

    Assumes all values are discrete integer indices.
    """
    object_shape = int(row["object_shape"])
    object_color = int(row["object_color"])
    floor_color = int(row["floor_color"])
    wall_color = int(row["wall_color"])

    object_shape_weight = lambda x: (int(x) - 2) / 4.0
    object_color_weight = lambda x: (int(x) - 5) * 0.5
    floor_color_weight = lambda x: (10 - int(x)) * 0.75
    wall_color_weight = lambda x: int(x) / 10

    raw_reward = (
        object_shape_weight(object_shape)
        + object_color_weight(object_color)
        + floor_color_weight(floor_color)
        + wall_color_weight(wall_color)
    )

    if quantize:
        if raw_reward >= 0.75:
            return 2.0
        elif raw_reward > 0:
            return 1.0
        elif raw_reward > -0.75:
            return -1.0
        else:
            return -2.0

    return raw_reward


# ---------------------------------------------------------------------
# Returned factor container
# ---------------------------------------------------------------------

@dataclass
class Shapes3DFactors:
    object_shape: torch.Tensor
    object_color: torch.Tensor
    floor_color: torch.Tensor
    wall_color: torch.Tensor

    def to(self, device: torch.device):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, value.to(device))
        return self
    
    def todict(self):
        return self.__dict__


@dataclass
class Input:
    pixel_values: torch.FloatTensor
    factors: Shapes3DFactors
    rewards: torch.FloatTensor

    def to(self, device: torch.device):
        self.pixel_values = self.pixel_values.to(device)
        self.factors = self.factors.to(device)
        self.rewards = self.rewards.to(device)
        return self
    

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class Shapes3DDataset(Dataset):
    """
    Simplified Shapes3D dataset.

    It always returns only:
        object_shape
        object_color
        floor_color
        wall_color

    The user can filter any of these factors. Any omitted selector means:
    keep all values for that factor.

    Scale and orientation are not returned, and are never filtered unless you
    explicitly add that later. Therefore, they are naturally marginalized over.
    """

    FACTOR_MAP = {
        "floor_color": 0,
        "wall_color": 1,
        "object_color": 2,
        "object_scale": 3,
        "object_shape": 4,
        "object_orientation": 5,
    }

    RETURNED_FACTORS = [
        "object_shape",
        "object_color",
        "floor_color",
        "wall_color",
    ]

    def __init__(
        self,
        file_path: str,
        object_shape: Iterable[Shape | str | int] | None = None,
        object_color: Iterable[HueColor | str | int] | None = None,
        floor_color: Iterable[HueColor | str | int] | None = None,
        wall_color: Iterable[HueColor | str | int] | None = None,
        train: bool = True,
        split_frac: float = 0.8,
        random_state: int = 44,
        resize: int | None = None,
        reward_fn: RewardFn | None = None,
    ):
        self.file_path = file_path
        self.resize = resize
        self.reward_fn = reward_fn

        self._h5 = None
        self.images = None

        with h5py.File(self.file_path, "r") as f:
            raw_labels = f["labels"][:]
            self.num_total = raw_labels.shape[0]
            self.images = f["images"][:]

        # Convert raw Shapes3D labels to discrete integer indices.
        df_dict = {"h5_idx": np.arange(self.num_total)}

        for name, col_idx in self.FACTOR_MAP.items():
            unique_vals = np.unique(raw_labels[:, col_idx])
            val_to_idx = {val: i for i, val in enumerate(unique_vals)}
            df_dict[name] = np.array([val_to_idx[v] for v in raw_labels[:, col_idx]])

        full_df = pd.DataFrame(df_dict)

        # Apply subpartition filters.
        filters = {
            "object_shape": object_shape,
            "object_color": object_color,
            "floor_color": floor_color,
            "wall_color": wall_color,
        }

        mask = np.ones(len(full_df), dtype=bool)

        for factor_name, allowed_values in filters.items():
            if allowed_values is not None:
                mask &= full_df[factor_name].isin(allowed_values).to_numpy()

        full_df = full_df.loc[mask].reset_index(drop=True)

        if len(full_df) == 0:
            raise ValueError(
                "The selected Shapes3D subpartition is empty. "
                "Check your selected shapes/colors."
            )

        # Remap global enum indices to local subset labels.
        self.global_to_local = {}
        self.local_to_global = {}

        for factor_name in self.RETURNED_FACTORS:
            global_values = sorted(full_df[factor_name].unique().tolist())

            global_to_local = {
                global_value: local_idx
                for local_idx, global_value in enumerate(global_values)
            }

            local_to_global = {
                local_idx: global_value
                for global_value, local_idx in global_to_local.items()
            }

            self.global_to_local[factor_name] = global_to_local
            self.local_to_global[factor_name] = local_to_global

            full_df[f"{factor_name}_global"] = full_df[factor_name]
            full_df[factor_name] = full_df[factor_name].map(global_to_local)

        # Add reward.
        if reward_fn is None:
            full_df["reward"] = 0.0
        else:
            full_df["reward"] = full_df.apply(reward_fn, axis=1).astype(np.float32)

        # Stratify by the four returned factors.
        # This preserves the selected combinations across train/test.
        stratify_key = full_df[self.RETURNED_FACTORS].apply(
            lambda x: "_".join(x.values.astype(str)),
            axis=1,
        )

        train_df, test_df = train_test_split(
            full_df,
            train_size=split_frac,
            stratify=stratify_key,
            random_state=random_state,
        )

        self.data_table = (
            train_df.reset_index(drop=True)
            if train
            else test_df.reset_index(drop=True)
        )

    def __len__(self):
        return len(self.data_table)

    def _get_images_handle(self):
        """
        Lazily open HDF5 file.

        This is safer with DataLoader workers than keeping an h5py handle
        opened before multiprocessing starts.
        """
        if self.images is not None:
            return self.images

        if self._h5 is None:
            self._h5 = h5py.File(self.file_path, "r")

        return self._h5["images"]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Shapes3DFactors, torch.Tensor]:
        sample = self.data_table.iloc[idx]

        images = self._get_images_handle()
        img = images[int(sample.h5_idx)]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.resize is not None:
            img = F.resize(img, (self.resize, self.resize))

        factors = Shapes3DFactors(
            object_shape=torch.tensor(int(sample["object_shape"]), dtype=torch.long),
            object_color=torch.tensor(int(sample["object_color"]), dtype=torch.long),
            floor_color=torch.tensor(int(sample["floor_color"]), dtype=torch.long),
            wall_color=torch.tensor(int(sample["wall_color"]), dtype=torch.long),
        )

        reward = torch.tensor(float(sample["reward"]), dtype=torch.float32)

        return img, factors, reward

    def __getstate__(self):
        """
        Prevent h5py file handles from being pickled by DataLoader workers.
        """
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def get_factor_sizes(self) -> dict[str, int]:
        return {
            name: int(self.data_table[name].nunique())
            for name in self.RETURNED_FACTORS
        }

    def get_selected_values(self) -> dict[str, list[int]]:
        return {
            name: sorted(self.data_table[name].unique().tolist())
            for name in self.RETURNED_FACTORS
        }

    def get_num_combinations(self) -> int:
        return (
            self.data_table[self.RETURNED_FACTORS]
            .drop_duplicates()
            .shape[0]
        )

    def get_reward_summary(self) -> pd.Series:
        return self.data_table["reward"].describe()

    def get_feature_utilities(self, feature: str) -> pd.Series:
        if feature not in self.RETURNED_FACTORS:
            raise ValueError(
                f"Unknown feature '{feature}'. "
                f"Use one of {self.RETURNED_FACTORS}"
            )

        return self.data_table.groupby(feature)["reward"].sum()


# ---------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------

def shapes3d_collate_fn(batch):
    imgs, factors_list, rewards = zip(*batch)

    batch_kwargs = {}

    for field in fields(Shapes3DFactors):
        batch_kwargs[field.name] = torch.stack([
            getattr(factors, field.name)
            for factors in factors_list
        ])

    return Input(
        pixel_values=torch.stack(imgs),
        factors=Shapes3DFactors(**batch_kwargs),
        rewards=torch.stack(rewards),
    )


# ---------------------------------------------------------------------
# Controller functions
# ---------------------------------------------------------------------

def get_Shapes3D_datasets(
    file_path: str,
    object_shape: Iterable[Shape | str | int] | None = None,
    object_color: Iterable[HueColor | str | int] | None = None,
    floor_color: Iterable[HueColor | str | int] | None = None,
    wall_color: Iterable[HueColor | str | int] | None = None,
    split_frac: float = 0.8,
    random_state: int = 44,
    resize: int | None = None,
    reward_fn: RewardFn | None = None,
) -> Tuple[Shapes3DDataset, Shapes3DDataset]:

    train_ds = Shapes3DDataset(
        file_path=file_path,
        object_shape=object_shape,
        object_color=object_color,
        floor_color=floor_color,
        wall_color=wall_color,
        train=True,
        split_frac=split_frac,
        random_state=random_state,
        resize=resize,
        reward_fn=reward_fn
    )

    test_ds = Shapes3DDataset(
        file_path=file_path,
        object_shape=object_shape,
        object_color=object_color,
        floor_color=floor_color,
        wall_color=wall_color,
        train=False,
        split_frac=split_frac,
        random_state=random_state,
        resize=resize,
        reward_fn=reward_fn,
    )
    
    reward_values = np.sort(list(set(train_ds.data_table.reward.unique()).union(
        set(test_ds.data_table.reward.unique())
    )))

    return train_ds, test_ds, reward_values


def get_Shapes3D_dataloaders(
    train_ds, test_ds,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    **dataloader_kwargs,
) -> Tuple[DataLoader, DataLoader]:

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=shapes3d_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataloader_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=shapes3d_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataloader_kwargs,
    )

    return train_loader, test_loader