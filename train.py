import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # For better plot aesthetics
import torch

from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
    VISUALIZATION_PATH,
    criterion,
    device,
    model,
    optimizer,
    scheduler,
)
from model import RegDGCNN
from utils import read_assets


sns.set_theme("whitegrid")

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

cd_targets = np.array(cd_targets).reshape(-1, 1)
scaler = MinMaxScaler()
cd_targets = scaler.fit_transform(cd_targets).flatten()

joblib.dump(scaler, MODELS_PATH / "scaler.joblib")

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

train_losses = []
val_losses = []
learning_rates = []

print("Starting training...")

try:
    for epoch in range(EPOCHS_COUNT):
        model.train()
        total_train_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            output, embedding = model(data, return_embedding=True)
            regression_loss = criterion(output, torch.sigmoid(data.y.unsqueeze(1)))

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

        if scheduler:
            scheduler.step(avg_val_loss)
            current_lr = scheduler.get_last_lr()[0]
            learning_rates.append(current_lr)
        else:
            learning_rates.append(optimizer.param_groups[0]["lr"])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), MODELS_PATH / "best_model.pth")
        else:
            trigger_times += 1
            if trigger_times >= STOP_LOSS_PATIENCE:
                print("Early stopping!")
                break

except KeyboardInterrupt:
    print("\nTraining interrupted by user!")

finally:
    print("Proceeding to evaluation and plotting...")


model.load_state_dict(torch.load(MODELS_PATH / "best_model.pth", weights_only=True))


try:
    model.eval()
    total_test_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)

            print(f"Output Shape: {output.shape}")
            print(f"Target Shape: {data.y.shape}")

            loss = criterion(output, data.y.unsqueeze(1))
            total_test_loss += loss.item() * data.num_graphs
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(output.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.6f}")

    y_true = scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"Test MAPE: {mape:.2f}%")
    print(f"Test R² Score: {r2:.4f}")

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predictions vs True Values")
    plt.plot(
        [min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--", label="Perfect Prediction"
    )
    corr_coef = np.corrcoef(y_true, y_pred)[0, 1]
    plt.text(
        0.05,
        0.95,
        f"Correlation: {corr_coef:.2f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALIZATION_PATH / "predictions_vs_true.png")
    plt.show()

    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.xlabel("Residuals")
    plt.title("Histogram of Residuals")
    plt.tight_layout()
    plt.savefig(VISUALIZATION_PATH / "residuals_histogram.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")
    plt.tight_layout()
    plt.savefig(VISUALIZATION_PATH / "residuals_vs_predicted.png")
    plt.show()

    errors = np.abs(y_true - y_pred)
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=50, kde=True, color="orange")
    plt.xlabel("Absolute Error")
    plt.title("Histogram of Absolute Errors")
    plt.tight_layout()
    plt.savefig(VISUALIZATION_PATH / "absolute_errors_histogram.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, label="Training Loss", marker="o")
    plt.plot(epochs_range, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(VISUALIZATION_PATH / "training_validation_loss.png")
    plt.show()

    if scheduler and len(learning_rates) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, learning_rates, label="Learning Rate", marker="x", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(VISUALIZATION_PATH / "learning_rate_schedule.png")
        plt.show()

except KeyboardInterrupt:
    print("Evaluation interrupted!")
