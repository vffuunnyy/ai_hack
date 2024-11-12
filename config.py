from pathlib import Path

import torch

from torch import nn

from model import RegDGCNN
from utils import Settings


MODELS_PATH = Path("./models")
ASSETS_PATH = Path("./assets")
RESULTS_FILE = "data.csv"

EPOCHS_COUNT = 10_000
MODELS_COUNT_LIMIT = None
BATCH_SIZE = 16
STOP_LOSS_PATIENCE = 256
BETA = 0.05

settings = Settings(
    points_range=range(1024, 4096 + 1, 256),
    assets_path=ASSETS_PATH,
    data_file=RESULTS_FILE,
    limit=MODELS_COUNT_LIMIT,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegDGCNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.MSELoss()
