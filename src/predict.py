"""Inference from a saved checkpoint.

Deliberately kept out of evaluate.py: that module holds pure scikit-learn
metric helpers and is imported by baselines.py, which has nothing to do with
torch and must not pull in the model definition.

"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, LABELS_INVERSE
from src.data import ChestXrayDataset, val_chain
from src.model import Model


def load_checkpoint(checkpoint_path, model_class=Model, device=None, verbose=True):
    """Rebuild a trained model from a checkpoint written by src/train.py.

    Only the state_dict was saved, so the architecture has to be supplied by
    the caller. Passing the wrong model_class raises here, on the key names,
    rather than producing silently wrong predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = model_class().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if verbose:
        print(
            f"loaded {Path(checkpoint_path).name}"
            f" | run_name: {checkpoint.get('run_name')}"
            f" | epoch: {checkpoint.get('epoch')}"
            f" | val_bal_acc at save: {checkpoint.get('val_bal_acc'):.4f}"
        )

    return model, device


def predict(df, checkpoint_path, model_class=Model, batch_size=BATCH_SIZE,
            num_workers=0, device=None, verbose=True):
    """Predict on every row of df, in row order.

    Returns (y_true, y_pred, prob_pneumonia) as plain lists aligned with the
    positional order of df, so a subset can be selected afterwards by index.

    Labels come back as strings, matching the label space used by
    src/evaluate.py and by the baselines already logged in experiments/runs.csv.

    val_chain is used unconditionally. Augmentation belongs to training only:
    evaluating through a stochastic pipeline would make the score
    irreproducible and would not measure the model on the data it will see.

    num_workers defaults to 0 because this is called from notebooks, where
    spawning worker processes on Windows re-imports the parent module.
    """
    model, device = load_checkpoint(checkpoint_path, model_class, device, verbose)

    dataset = ChestXrayDataset(df, val_chain)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    preds, labels, probs = [], [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            outputs = model(images.to(device))

            batch_probs = torch.softmax(outputs, dim=1)[:, 1]

            preds.append(outputs.argmax(dim=1).cpu())
            probs.append(batch_probs.cpu())
            labels.append(batch_labels)

    y_pred = [LABELS_INVERSE[p] for p in torch.cat(preds).tolist()]
    y_true = [LABELS_INVERSE[i] for i in torch.cat(labels).tolist()]
    prob_pneumonia = torch.cat(probs).tolist()

    return y_true, y_pred, prob_pneumonia
