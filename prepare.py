from pathlib import Path

import numpy as np
import pandas as pd

from stl import mesh


source_dir = Path("input")
output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = Path("data.csv")
output_csv_path = output_dir / "output.csv"

data = pd.read_csv(csv_path)

data['Design'] = data['Design'] + ".stl"

data[['Design', 'Average Cd']].to_csv(output_csv_path, index=False)
print(f"Новый файл CSV сохранён: {output_csv_path}")

def combine_stl_files(folder: Path, output_path: Path) -> None:
    stl_files = list(folder.glob("*.stl"))
    if len(stl_files) < 2:
        print(f"Недостаточно STL-файлов в папке {folder}, пропуск...")
        return

    combined_data = []
    for stl_file in stl_files:
        print(f"Загрузка: {stl_file}")
        model = mesh.Mesh.from_file(str(stl_file))
        combined_data.append(model.data)

    combined_mesh = mesh.Mesh(np.concatenate(combined_data))

    combined_mesh.save(output_path)
    print(f"Объединённый файл сохранён: {output_path}")

for folder in source_dir.iterdir():
    if folder.is_dir():
        output_file = output_dir / f"{folder.name}.stl"
        print(f"Обработка папки: {folder}")
        combine_stl_files(folder, output_file)

print("Обработка завершена.")
