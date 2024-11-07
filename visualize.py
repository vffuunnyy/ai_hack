import matplotlib.pyplot as plt

from utils import Settings, read_assets


settings = Settings(
    points_range=range(2**10, 2**12 + 1, 2**9),
    z_cut_percent=0.1,
    assets_path="./assets",
    data_file="data.csv",
    limit=1,
)
point_clouds, normals, cd_targets = read_assets(settings)

print(f"Number of models: {len(point_clouds)}")


def plot_point_cloud(point_cloud, normals, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c="r", marker="o")
    ax.quiver(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        normals[:, 0],
        normals[:, 1],
        normals[:, 2],
    )
    ax.set_title(title)
    plt.show()


for points, norms in zip(point_clouds, normals):
    plot_point_cloud(points, norms, title=f"Point cloud with {len(points)} points")
