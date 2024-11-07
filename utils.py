from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from stl.mesh import Mesh
from tqdm import tqdm


@dataclass
class Settings:
    """Settings for the dataset."""

    points_range: range = field(default_factory=lambda: range(1024, 4096, 256))
    z_cut: float = 0.1
    z_cut_percent: float = 0.1
    assets_path: str = "./assets"
    data_file: str = "data.csv"
    limit: int | None = None


def resample_points_mbkmeans(points: np.array, points_num: int = 2048) -> np.array:
    """Resamples the points using Mini-Batch K-Means clustering with KMeans++ initialization.

    Optionally applies PCA for dimensionality reduction before clustering.

    Args:
        points (np.array): Input points.
        points_num (int, optional): Number of clusters. Defaults to 2048.
        use_pca (bool, optional): If True, applies PCA before clustering. Defaults to False.
        n_components (int, optional): Number of PCA components. If None, retains all components.

    Returns:
        np.array: Resampled points (cluster centers).
    """

    mbkmeans = MiniBatchKMeans(
        n_clusters=points_num, init="k-means++", batch_size=512, max_iter=10, random_state=42
    )
    mbkmeans.fit(points)
    return mbkmeans.cluster_centers_


def stl_to_point_cloud(filename: str, z_cut: float = 0.1) -> tuple[np.array, np.array]:
    """Converts an STL file to a point cloud and extracts normals.

    Args:
        filename (str): Path to the STL file.
        z_cut (float, optional): Defaults to 0.1.

    Returns:
        tuple[np.array, np.array]: Point cloud and normals.
    """
    mesh_data = Mesh.from_file(filename)
    points = mesh_data.vectors.reshape(-1, 3)
    normals = np.repeat(mesh_data.normals, 3, axis=0)

    assert (  # noqa: S101
        len(points) == len(normals)
    ), f"Points and normals must match in length, but got {len(points)} points and {len(normals)} normals."

    # Убираем дублирующиеся точки и нормали вместе
    unique_points, indices = np.unique(points, axis=0, return_index=True)
    unique_normals = normals[indices]

    # Обрезаем по z_cut
    min_z, max_z = unique_points[:, 2].min(), unique_points[:, 2].max()
    z_cut_threshold = min_z + z_cut * (max_z - min_z)
    mask = unique_points[:, 2] >= z_cut_threshold
    return unique_points[mask], unique_normals[mask]  # Применяем ту же маску к точкам и нормалям


def process_design(
    design_name: str, cd_value: float, settings: Settings
) -> tuple[list[np.array], list[np.array], list[float]]:
    """Processes a design to extract point clouds and normals, and associates CD targets.

    Args:
        design_name (str): Design name.
        cd_value (float): CD target value.
        settings (Settings): Settings object.

    Returns:
        tuple[list[np.array], list[np.array], list[float]]: Point clouds, normals, and CD targets.
    """
    filename = f"{settings.assets_path}/{design_name}.stl"
    points_base, normals_base = stl_to_point_cloud(filename, settings.z_cut_percent)
    point_clouds = []
    normals = []
    cd_targets = []

    for size in tqdm(
        settings.points_range, desc=f"Processing {design_name}", total=len(settings.points_range)
    ):
        points = resample_points_mbkmeans(points_base, size)
        resampled_indices = np.random.choice(len(points_base), size, replace=False)  # noqa: NPY002
        resampled_normals = normals_base[resampled_indices]

        point_clouds.append(points)
        normals.append(resampled_normals)
        cd_targets.append(cd_value)

    return point_clouds, normals, cd_targets


def read_assets(settings: Settings) -> tuple[list[np.array], list[np.array], list[float]]:
    """Reads the assets and extracts point clouds, normals, and CD targets.

    Args:
        settings (Settings): Settings object.

    Returns:
        tuple[list, list, list]: Point clouds, normals, and CD targets.
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
