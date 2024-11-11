from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stl_reducer import reduce_stl_points
from torch_geometric.data import Data
from tqdm import tqdm


@dataclass
class Settings:
    """Настройки для датасета."""

    points_range: range = field(default_factory=lambda: range(1024, 4096 + 1, 256))
    assets_path: str = "./assets"
    data_file: str = "data.csv"
    limit: int | None = None


def process_design(
    design_name: str, cd_value: float, settings: Settings
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Обрабатывает дизайн для извлечения облаков точек и нормалей, связывая их со значениями CD.

    Args:
        design_name (str): Имя дизайна.
        cd_value (float): Целевое значение CD.
        settings (Settings): Объект настроек.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[float]]: Облака точек, нормали и значения CD.
    """
    filename = Path(f"{settings.assets_path}/{design_name}").with_suffix(".stl")
    point_clouds = []
    normals = []
    cd_targets = []

    for size in tqdm(
        settings.points_range, desc=f"Processing {design_name}", total=len(settings.points_range)
    ):
        points, normals_output = reduce_stl_points(filename, size)
        point_clouds.append(np.array(points))
        normals.append(np.array(normals_output))
        cd_targets.append(cd_value)

    return point_clouds, normals, cd_targets


def read_assets(settings: Settings) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Считывает ресурсы и извлекает облака точек, нормали и значения CD.

    Args:
        settings (Settings): Объект настроек.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[float]]: Облака точек, нормали и значения CD.
    """
    data = pd.read_csv(
        f"{settings.assets_path}/{settings.data_file}", usecols=["Design", "Average Cd"]
    )
    if settings.limit:
        data = data.head(settings.limit)
    cd_values = dict(zip(data["Design"], data["Average Cd"]))

    args = [(design_name, cd_values[design_name], settings) for design_name in cd_values]

    point_clouds = []
    normals = []
    cd_targets = []

    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(process_design, args)

    for result in results:
        point_clouds.extend(result[0])
        normals.extend(result[1])
        cd_targets.extend(result[2])

    return point_clouds, normals, cd_targets


def random_rotate_point_cloud(data: Data) -> Data:
    """Вращает облако точек случайным образом.

    Args:
        data (Data): Объект Data с облаком точек.

    Returns: Объект Data с повернутым облаком точек.
    """
    theta = np.random.uniform(0, 2 * np.pi)  # noqa: NPY002
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    data.pos = torch.matmul(data.pos, torch.tensor(rotation_matrix, dtype=torch.float))
    data.normals = torch.matmul(data.normals, torch.tensor(rotation_matrix, dtype=torch.float))
    return data
