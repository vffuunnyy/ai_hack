import numpy as np
import pandas as pd
import torch

from mesh_reducer import load_meshes_range_points
from torch_geometric.data import Data

from config import ASSETS_PATH, POINTS_RANGE, READ_ASSETS_LIMIT, RESULTS_FILE


def read_assets() -> tuple[list[tuple[int, int, int]], list[float]]:
    """Reads assets and extracts point clouds and CD values.

    Args:
        settings (Settings): Settings object.

    Returns:
        Tuple[List[np.ndarray], List[float]]: Point clouds and CD values.
    """
    data = pd.read_csv(
        RESULTS_FILE.as_posix(),
        usecols=["Design", "Average Cd"],
    )
    if READ_ASSETS_LIMIT:
        data = data.head(READ_ASSETS_LIMIT)
    cd_values = dict(zip(data["Design"], data["Average Cd"]))

    file_paths = [f"{(ASSETS_PATH / file).as_posix()}" for file in cd_values]
    point_clouds, cd_targets = zip(*[
        (mesh.points, cd_values[mesh.name])
        for mesh in load_meshes_range_points(file_paths, POINTS_RANGE)
    ])
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
