import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from config import (
    ASSETS_COUNT,
    ASSETS_PATH,
    BATCH_SIZE,
    BETA,
    EPOCHS_COUNT,
    LOAD_BEST_MODEL,
    MODELS_PATH,
    POINTS_RANGE,
    READ_ASSETS_LIMIT,
    RESULTS_FILE,
    STOP_LOSS_PATIENCE,
    criterion,
    device,
    model,
    optimizer,
    scheduler,
)
from model import RegDGCNN
from utils import random_rotate_point_cloud, read_assets


print(
    "Current settings:\n"
    "\n"
    "Settings for the model:\n"
    f"Device: {device}\n"
    f"Model: {model}\n"
    f"Optimizer: {optimizer}\n"
    f"Scheduler: {scheduler}\n"
    f"Criterion: {criterion}\n"
    "\n"
    "Train settings:\n"
    f"Epochs: {EPOCHS_COUNT}\n"
    f"Batch Size: {BATCH_SIZE}\n"
    f"Stop Loss Patience: {STOP_LOSS_PATIENCE}\n"
    f"Beta: {BETA}\n"
    f"Best model loading: {LOAD_BEST_MODEL}\n"
    "\n"
    "Settings for the dataset:\n"
    f"Points Range: {POINTS_RANGE}\n"
    f"Models Count with range: {ASSETS_COUNT}\n"
    f"Models Count limit: {READ_ASSETS_LIMIT}\n"
    f"3D Models path: {ASSETS_PATH.as_posix()}\n"
    f"Train Results file: {RESULTS_FILE.as_posix()}\n"
    f"Models path: {MODELS_PATH.as_posix()}\n"
    "\n"
    "Loading models..."
)

point_clouds, cd_targets = read_assets()

print("3D Models loaded")
print(f"Number of 3D models: {len(point_clouds)}")

os.makedirs(MODELS_PATH, exist_ok=True)  # noqa: PTH103

cd_targets = np.array(cd_targets).reshape(-1, 1)
scaler = StandardScaler()
cd_targets = scaler.fit_transform(cd_targets).flatten()

data_list = []
for points, cd in zip(point_clouds, cd_targets):
    pos = torch.tensor(points, dtype=torch.float)
    y = torch.tensor([cd], dtype=torch.float)
    data_list.append(Data(pos=pos, y=y))

train_val_data, test_data = train_test_split(data_list, test_size=0.15, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.15 / 0.85, random_state=42)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

prior_network = RegDGCNN().to(device)
prior_network.eval()
for param in prior_network.parameters():
    param.requires_grad = False

best_val_loss = float("inf")
trigger_times = 0
rng = np.random.default_rng()

train_losses = []
val_losses = []

print("Strting training...")

try:
    for epoch in range(EPOCHS_COUNT):
        model.train()
        total_train_loss = 0

        for data in train_loader:
            data = random_rotate_point_cloud(rng, data)
            data = data.to(device)
            optimizer.zero_grad()

            output, embedding = model(data, return_embedding=True)
            regression_loss = criterion(output, data.y.unsqueeze(1))

            with torch.no_grad():
                _, prior_embedding = prior_network(data, return_embedding=True)

            rnd_loss = nn.functional.mse_loss(embedding, prior_embedding)
            total_loss = regression_loss + BETA * rnd_loss

            total_loss.backward()
            optimizer.step()

            total_train_loss += regression_loss.item() * data.num_graphs

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS_COUNT}, Training Loss: {avg_train_loss:.6f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output, embedding = model(data, return_embedding=True)

                regression_loss = criterion(output, data.y.unsqueeze(1))
                _, prior_embedding = prior_network(data, return_embedding=True)

                rnd_loss = nn.functional.mse_loss(embedding, prior_embedding)
                total_loss = regression_loss + BETA * rnd_loss
                total_val_loss += total_loss.item() * data.num_graphs

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS_COUNT}, Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), "./models/best_model.pth")
        else:
            trigger_times += 1
            if trigger_times >= STOP_LOSS_PATIENCE:
                print("Early stopping!")
                break

        scheduler.step()
except KeyboardInterrupt:
    print("Training interrupted!")

model.load_state_dict(torch.load("./models/best_model.pth", weights_only=True))

try:
    model.eval()
    total_test_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y.unsqueeze(1))
            total_test_loss += loss.item() * data.num_graphs
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(output.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.6f}")

    y_true = scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"Test MAPE: {mape:.2f}%")

    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predictions vs True Values")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--")
    plt.grid(True)
    plt.savefig("./models/predictions.png")
    plt.show()
except KeyboardInterrupt:
    print("Evaluation interrupted!")

plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("./models/loss_plot.png")
plt.show()

torch.save(model.state_dict(), "./models/final_model.pth")
