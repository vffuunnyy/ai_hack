import argparse

from pathlib import Path

from mesh_reducer import MeshObject, load_meshes


def load_3d_models(path: Path, points_count: int) -> list[MeshObject]:
    models = [file for file in path.iterdir() if file.suffix in [".stl", ".obj", ".ply"]]
    return load_meshes(models, points_count)


def evaluate_model(
    points_count: int, assets_path: Path, model_file: Path, output_file: Path
) -> None:
    from csv import DictWriter

    import torch

    from torch_geometric.data import Data

    from model import RegDGCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegDGCNN().to(device)
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()

    meshes = load_3d_models(assets_path, points_count)

    writer = None
    if output_file:
        file = Path(output_file).open("w", newline="")  # noqa: SIM115
        writer = DictWriter(file, fieldnames=["Design", "Average Cd"])
        writer.writeheader()

    with torch.no_grad():
        for mesh in meshes:
            points_tensor = torch.tensor(mesh.points, dtype=torch.float32).to(device)
            data = Data(pos=points_tensor)
            predictions = model(data).item()

            print(f"Evaluating {mesh.name}: {predictions}")

            if writer:
                writer.writerow({"Design": mesh.name, "Average Cd": predictions})

    if writer:
        file.close()

    print("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D models with a neural network.")
    parser.add_argument(
        "--points", type=int, required=False, default=512, help="Number of points to reduce to."
    )
    parser.add_argument(
        "--assets",
        type=str,
        required=True,
        help="Path to the directory containing 3D models in STL, OBJ, or PLY format.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the neural network model file."
    )
    parser.add_argument("--output", type=str, required=False, help="Path to the output CSV file.")

    args = parser.parse_args()
    evaluate_model(
        args.points, Path(args.assets), Path(args.model), Path(args.output) if args.output else None
    )
