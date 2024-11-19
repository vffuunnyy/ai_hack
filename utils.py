from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from mesh_reducer import load_meshes_range_points
from torch_geometric.data import Data


@dataclass
class Settings:
    """Settings for the dataset."""

    points_range: range = field(default_factory=lambda: range(1024, 4096 + 1, 256))
    assets_path: str = "./assets"
    data_file: str = "data.csv"
    limit: int | None = None


def read_assets(settings: Settings) -> tuple[list[tuple[int, int, int]], list[float]]:
    """Reads assets and extracts point clouds and CD values.

    Args:
        settings (Settings): Settings object.

    Returns:
        Tuple[List[np.ndarray], List[float]]: Point clouds and CD values.
    """
    data = pd.read_csv(
        f"{settings.assets_path}/{settings.data_file}", usecols=["Design", "Average Cd"]
    )
    if settings.limit:
        data = data.head(settings.limit)
    cd_values = dict(zip(data["Design"], data["Average Cd"]))

    file_paths = [f"{settings.assets_path}/{file}" for file in cd_values]
    point_clouds, cd_targets = zip(*[
        (points, cd_values[design_name])
        for design_name, points in zip(
            cd_values, load_meshes_range_points(file_paths, list(settings.points_range))
        )
    ])

    print(len(point_clouds), len(cd_targets))

    return list(point_clouds), list(cd_targets)


def random_rotate_point_cloud(rng: np.random.Generator, data: Data) -> Data:
    """Randomly rotates the point cloud horizontally.

    Args:
        data (Data): Data object with point cloud.

    Returns: Data object with rotated point cloud.
    """
    theta = rng.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    data.pos[:, [0, 1]] = torch.matmul(
        data.pos[:, [0, 1]], torch.tensor(rotation_matrix[:2, :2], dtype=torch.float)
    )
    return data
