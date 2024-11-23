from pathlib import Path

import torch

from torch import nn

from model import RegDGCNN


LOAD_BEST_MODEL = False
USE_SEABORN = True

MODELS_PATH = Path("./models")
ASSETS_PATH = Path("./assets")
VISUALIZATION_PATH = Path("./visualization")
RESULTS_FILE = ASSETS_PATH / "data.csv"

MODELS_PATH.mkdir(parents=True, exist_ok=True)
VISUALIZATION_PATH.mkdir(parents=True, exist_ok=True)

POINTS_RANGE = list(range(1024, 4096 + 1, 256))
READ_ASSETS_LIMIT = None

EPOCHS_COUNT = 10_000
BATCH_SIZE = 32
STOP_LOSS_PATIENCE = 100
BETA = 0.01

ASSETS_COUNT = len(list(ASSETS_PATH.glob("*.stl"))) * len(POINTS_RANGE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegDGCNN().to(device)

if LOAD_BEST_MODEL and (best_model := (MODELS_PATH / "best_model.pth")).exists():
    model.load_state_dict(torch.load(best_model))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, verbose=True
)
criterion = nn.MSELoss()
