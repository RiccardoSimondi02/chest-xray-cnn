"""Training loop for the baseline CNN.

Deliberately minimal: no batch normalisation, no dropout, no augmentation, no
learning-rate schedule and no early stopping. 

"""

import csv
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.config import LABELS_INVERSE, N_EPOCHS
from src.data import ChestXrayDataset, return_split, train_chain, val_chain
from src.evaluate import return_balanced_accuracy_score, return_confusion_matrix
from src.model import Model

RUN_NAME = "cnn_baseline"
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
SEED = 42

HISTORY_CSV = Path("experiments") / f"history_{RUN_NAME}.csv"
CKPT_DIR = Path("experiments") / "checkpoints"


if __name__ == "__main__":

    # --- device -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(SEED)

    # --- data -------------------------------------------------------------
    train_df, val_df, _ = return_split()

    train_dataset = ChestXrayDataset(train_df, train_chain)
    val_dataset = ChestXrayDataset(val_df, val_chain)

    # persistent_workers keeps the worker processes alive between epochs;
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, persistent_workers=True,
        pin_memory=(device.type == "cuda"),
    )
    # shuffle=False on validation: predictions then come out in the same order as the rows of val_df
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, persistent_workers=True,
        pin_memory=(device.type == "cuda"),
    )

    # --- model, loss, optimiser -------------------------------------------
    # The model is moved to the device BEFORE the optimiser is built, so that
    # the optimiser holds references to the parameters that are actually used.
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"run: {RUN_NAME} | parameters: {n_params:,} | epochs: {N_EPOCHS}")

    # --- run artefacts ----------------------------------------------------
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_CSV, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_bal_acc", "seconds"]
        )

    best_bal_acc = -1.0

    # --- epochs -----------------------------------------------------------
    for epoch in range(N_EPOCHS):
        epoch_start = time.perf_counter()

        # training phase
        model.train()
        total_loss_train = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item() * len(labels)

        train_mean_loss = total_loss_train / len(train_dataset)

        # validation phase: no zero_grad, no backward, no step
        model.eval()
        total_loss_val = 0.0
        total_preds = []
        total_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss_val += loss.item() * len(labels)

                total_preds.append(outputs.argmax(dim=1).cpu())
                total_labels.append(labels.cpu())

        val_mean_loss = total_loss_val / len(val_dataset)

        total_preds = torch.cat(total_preds)
        total_labels = torch.cat(total_labels)

        # Back to label strings: the baselines in experiments/runs.csv were
        # scored on the same label space, so the numbers stay comparable.
        preds_str = [LABELS_INVERSE[p] for p in total_preds.tolist()]
        labels_str = [LABELS_INVERSE[l] for l in total_labels.tolist()]

        bal_acc_score_val = return_balanced_accuracy_score(labels_str, preds_str)
        conf_matrix_val = return_confusion_matrix(labels_str, preds_str)

        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch + 1}/{N_EPOCHS}"
            f" - train_loss: {train_mean_loss:.4f}"
            f" - val_loss: {val_mean_loss:.4f}"
            f" - bal_acc: {bal_acc_score_val:.4f}"
            f" - time: {epoch_time:.2f}s"
        )

        with open(HISTORY_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                round(train_mean_loss, 6),
                round(val_mean_loss, 6),
                round(bal_acc_score_val, 6),
                round(epoch_time, 2),
            ])

        if bal_acc_score_val > best_bal_acc:
            best_bal_acc = bal_acc_score_val
            torch.save(
                {
                    "run_name": RUN_NAME,
                    "epoch": epoch + 1,
                    "val_bal_acc": bal_acc_score_val,
                    "model_state_dict": model.state_dict(),
                },
                CKPT_DIR / f"{RUN_NAME}_best.pt",
            )
            print(f"    new best ({bal_acc_score_val:.4f}) - checkpoint saved")

    # --- end of run -------------------------------------------------------
    print(f"\nbest validation balanced accuracy: {best_bal_acc:.4f}")
    print("confusion matrix at the last epoch (rows: true, cols: predicted)")
    print(f"           {'NORMAL':>10} {'PNEUMONIA':>10}")
    for name, row in zip(["NORMAL", "PNEUMONIA"], conf_matrix_val):
        print(f"{name:>10} {row[0]:>10} {row[1]:>10}")
