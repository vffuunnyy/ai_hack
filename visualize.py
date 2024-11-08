import numpy as np
import pyvista as pv

from utils import Settings, read_assets


settings = Settings(
    points_range=range(2**12, 2**12 + 1, 1),
    assets_path="./assets",
    data_file="data.csv",
    limit=1,
)
point_clouds, normals, cd_targets = read_assets(settings)

print(f"Number of models: {len(point_clouds)}")


def plot_point_cloud(point_cloud: np.ndarray, normals: np.ndarray, title: str) -> None:
    try:
        cloud = pv.PolyData(point_cloud)
        cloud["normals"] = normals
        plotter = pv.Plotter()
        plotter.add_mesh(cloud, scalars="normals", point_size=30, render_points_as_spheres=True)
        plotter.add_title(title)
        plotter.show()
    except Exception as e:
        print(f"Ошибка визуализации: {e}")


for points, norms in zip(point_clouds, normals):
    plot_point_cloud(points, norms, title=f"Point cloud with {len(points)} points")
