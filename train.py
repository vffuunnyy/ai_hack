import argparse
import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from config import (
    ASSETS_COUNT,
    ASSETS_PATH,
    BATCH_SIZE,
    BETA,
    EPOCHS_COUNT,
    LOAD_BEST_MODEL,
    LOGS_PATH,
    MODELS_PATH,
    POINTS_RANGE,
    READ_ASSETS_LIMIT,
    RESULTS_FILE,
    STOP_LOSS_PATIENCE,
    USE_SEABORN,
    VISUALIZATION_PATH,
    criterion,
    device,
    model,
    optimizer,
    scheduler,
)
from model import RegDGCNN
from utils import read_assets


parser = argparse.ArgumentParser(description="Train RegDGCNN Model")
parser.add_argument(
    "--no-seaborn",
    action="store_true",
    help="Disable Seaborn for plotting and do not display plots.",
)
args = parser.parse_args()

if args.no_seaborn:
    USE_SEABORN = False

if USE_SEABORN:
    import seaborn as sns

    sns.set_theme("notebook", "whitegrid")
else:
    sns = None

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

writer = SummaryWriter(log_dir=LOGS_PATH)

writer.add_text("Settings/Beta", str(BETA))
writer.add_text("Settings/Epochs", str(EPOCHS_COUNT))
writer.add_text("Settings/Batch Size", str(BATCH_SIZE))
writer.add_text("Settings/Stop Loss Patience", str(STOP_LOSS_PATIENCE))
writer.add_text("Settings/Assets Count", str(ASSETS_COUNT))
writer.add_text("Settings/Models Count Limit", str(READ_ASSETS_LIMIT))
writer.add_text("Settings/Points Range Min", str(min(POINTS_RANGE)))
writer.add_text("Settings/Points Range Max", str(max(POINTS_RANGE)))
writer.add_text("Settings/Models Path", ASSETS_PATH.as_posix())
writer.add_text("Settings/Results File", RESULTS_FILE.as_posix())
writer.add_text("Settings/Models Path", MODELS_PATH.as_posix())

writer.add_scalar("Loss/train", 0, 0)
writer.add_scalar("Loss/validation", 0, 0)


print("Starting training...")

try:
    for epoch in range(EPOCHS_COUNT):
        model.train()
        total_train_loss = 0

        for data in train_loader:
            try:
                data = data.to(device)
                optimizer.zero_grad()

                output, embedding, _ = model(data, return_embedding=True)
                regression_loss = criterion(output, data.y.unsqueeze(1))

                with torch.no_grad():
                    _, prior_embedding, _ = prior_network(data, return_embedding=True)

                rnd_loss = nn.functional.mse_loss(embedding, prior_embedding)
                total_loss = regression_loss + BETA * rnd_loss

                total_loss.backward()
                optimizer.step()

                total_train_loss += regression_loss.item() * data.num_graphs

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("Произошла ошибка OOM. Пропуск текущего батча.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS_COUNT}, Training Loss: {avg_train_loss:.6f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                try:
                    data = data.to(device)
                    output, embedding, _ = model(data, return_embedding=True)

                    regression_loss = criterion(output, data.y.unsqueeze(1))
                    _, prior_embedding, _ = prior_network(data, return_embedding=True)

                    rnd_loss = nn.functional.mse_loss(embedding, prior_embedding)
                    total_loss = regression_loss + BETA * rnd_loss
                    total_val_loss += total_loss.item() * data.num_graphs

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("Произошла ошибка OOM во время валидации. Пропуск текущего батча.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS_COUNT}, Validation Loss: {avg_val_loss:.6f}")
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)

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


# for data in val_loader:
#     writer.add_graph(model, data.to(device))

for name, param in model.named_parameters():
    writer.add_histogram(f"{name}", param, epoch)
    if param.grad is not None:
        writer.add_histogram(f"{name}.grad", param.grad, epoch)

writer.close()
model.load_state_dict(torch.load(MODELS_PATH / "best_model.pth", weights_only=True))

train_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
(VISUALIZATION_PATH / train_time).mkdir(parents=True, exist_ok=True)

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

    def plot_and_save(fig, filename):
        fig.tight_layout()
        fig.savefig(VISUALIZATION_PATH / train_time / filename)
        if USE_SEABORN:
            plt.show()
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    if USE_SEABORN:
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, ax=ax)
    else:
        ax.scatter(y_true, y_pred, alpha=0.6)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predictions vs True Values")
    ax.plot(
        [min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--", label="Perfect Prediction"
    )
    corr_coef = np.corrcoef(y_true, y_pred)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {corr_coef:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    ax.legend()
    plot_and_save(fig, "predictions_vs_true.png")

    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    if USE_SEABORN:
        sns.histplot(residuals, bins=50, kde=True, ax=ax)
    else:
        ax.hist(residuals, bins=50, density=True, alpha=0.6, color="g")
        ax.set_ylabel("Density")
    ax.set_xlabel("Residuals")
    ax.set_title("Histogram of Residuals")
    plot_and_save(fig, "residuals_histogram.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    if USE_SEABORN:
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=ax)
    else:
        ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted Values")
    plot_and_save(fig, "residuals_vs_predicted.png")

    errors = np.abs(y_true - y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    if USE_SEABORN:
        sns.histplot(errors, bins=50, kde=True, color="orange", ax=ax)
    else:
        ax.hist(errors, bins=50, density=True, alpha=0.6, color="orange")
        ax.set_ylabel("Density")
    ax.set_xlabel("Absolute Error")
    ax.set_title("Histogram of Absolute Errors")
    plot_and_save(fig, "absolute_errors_histogram.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, label="Training Loss", marker="o")
    ax.plot(epochs_range, val_losses, label="Validation Loss", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Over Epochs")
    ax.legend()
    ax.grid(True)
    plot_and_save(fig, "training_validation_loss.png")

    if scheduler and len(learning_rates) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs_range, learning_rates, label="Learning Rate", marker="x", color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.legend()
        ax.grid(True)
        plot_and_save(fig, "learning_rate_schedule.png")

except KeyboardInterrupt:
    print("Evaluation interrupted!")
