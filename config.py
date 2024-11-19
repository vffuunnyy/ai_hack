from pathlib import Path

import torch

from torch import nn

from model import RegDGCNN


LOAD_BEST_MODEL = False

MODELS_PATH = Path("./models")
ASSETS_PATH = Path("./assets")
RESULTS_FILE = ASSETS_PATH / "test.csv"

POINTS_RANGE = list(range(1024, 4096 + 1, 256))
READ_ASSETS_LIMIT = None

EPOCHS_COUNT = 10_000
BATCH_SIZE = 16
STOP_LOSS_PATIENCE = 256
BETA = 0.05

ASSETS_COUNT = len(list(ASSETS_PATH.glob("*.stl"))) * len(POINTS_RANGE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegDGCNN().to(device)

if LOAD_BEST_MODEL and (best_model := (MODELS_PATH / "best_model.pth")).exists():
    model.load_state_dict(torch.load(best_model))

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.MSELoss()
