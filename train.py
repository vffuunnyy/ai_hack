import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import RegDGCNN
from utils import Settings, read_assets


EPOCHS_COUNT = 100
settings = Settings(
    points_range=range(2**10, 2**11 + 1, 2**8),
    z_cut_percent=0.1,
    assets_path="./assets",
    data_file="data.csv",
    limit=4,
)
point_clouds, cd_targets = read_assets(settings)

print(f"Number of models: {len(point_clouds)}")

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

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegDGCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
criterion = nn.SmoothL1Loss()

os.makedirs("./models", exist_ok=True)  # noqa: PTH103

best_val_loss = float("inf")
patience = 10
trigger_times = 0

train_losses = []
val_losses = []

try:
    for epoch in range(EPOCHS_COUNT):
        torch.cuda.empty_cache()
        model.train()
        total_train_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * data.num_graphs
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data.y.unsqueeze(1))
                total_val_loss += loss.item() * data.num_graphs
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS_COUNT}, "
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), "./models/best_model.pth")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break
except KeyboardInterrupt:
    print("Training interrupted!")

model.load_state_dict(torch.load("./models/best_model.pth"))

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
    print(f"Test Loss: {avg_test_loss:.4f}")

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
