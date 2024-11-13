# from pathlib import Path

# import numpy as np
# import pyvista as pv

# from utils import Settings, read_assets


# MODELS_PATH = Path("./models")
# ASSETS_PATH = Path("./assets")
# RESULTS_FILE = "data copy.csv"

# EPOCHS_COUNT = 10_000
# MODELS_COUNT_LIMIT = 10
# BATCH_SIZE = 8
# STOP_LOSS_PATIENCE = 256
# BETA = 0.05

# settings = Settings(
#     points_range=range(1024, 1024 + 1, 256),
#     assets_path=ASSETS_PATH,
#     data_file=RESULTS_FILE,
#     limit=MODELS_COUNT_LIMIT,
# )
# point_clouds, cd_targets = read_assets(settings)

# print(f"Number of models: {len(point_clouds)}")


# def plot_point_cloud(point_cloud: np.ndarray, title: str) -> None:
#     try:
#         cloud = pv.PolyData(point_cloud)
#         plotter = pv.Plotter()
#         plotter.add_mesh(cloud, point_size=30, render_points_as_spheres=True)
#         plotter.add_title(title)
#         plotter.show()
#     except Exception as e:
#         print(f"Ошибка визуализации: {e}")


# for points in point_clouds:
#     plot_point_cloud(points, title=f"Point cloud with {len(points)} points")


import pyvista as pv

from mesh_reducer import reduce_mesh_points


points = reduce_mesh_points("assets/mesh.obj", 10000)
cloud = pv.PolyData(points)
plotter = pv.Plotter()
plotter.add_mesh(cloud, point_size=20, render_points_as_spheres=True)
plotter.show()
