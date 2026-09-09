"""Final evaluation on the test set.

Two models are reported, decided before looking at any test figure: the network
trained from scratch and the fine-tuned ResNet18. 

Run once per model, from the project root:

    # 1. set MODEL = "cnn" in src/config.py
    python -m src.final_eval
    # 2. set MODEL = "resnet18_finetune" in src/config.py
    python -m src.final_eval

A separate process each time, so config.py is re-read; 

MODEL is the single switch: it selects the preprocessing (through val_chain),
the checkpoint and the architecture together, so they cannot disagree.
"""

from pathlib import Path

from src.config import MODEL
from src.data import return_split
from src.evaluate import return_balanced_accuracy_score, return_confusion_matrix
from src.model import Model, get_resnet_model
from src.predict import predict

CKPT_DIR = Path("experiments") / "checkpoints"

# MODEL -> (checkpoint file, callable building the matching empty architecture).
# A checkpoint stores only a state_dict, so the architecture has to be rebuilt
# here before the weights can be loaded into it.
FINAL_MODELS = {
    "cnn": (
        # run 006 / cnn_gap_80_s42, best epoch 55. The filename predates the
        # current naming scheme; predict() prints the run_name stored inside.
        "cnn_baseline_gap_80_epochs_best.pt",
        Model,
    ),
    "resnet18_finetune": (
        # run 016, best epoch 19
        "cnn_20_none_none_s42_resnet18_finetune_best.pt",
        lambda: get_resnet_model("resnet18_finetune"),
    ),
}


def report(name, y_true, y_pred):
    bal_acc = return_balanced_accuracy_score(y_true, y_pred)
    cm = return_confusion_matrix(y_true, y_pred)

    print(f"\n{name}  (n = {len(y_true)})")
    print(f"  balanced accuracy: {bal_acc:.4f}")
    print(f"  {'':>10} {'pred N':>8} {'pred P':>8}")
    for label, row in zip(["true N", "true P"], cm):
        print(f"  {label:>10} {row[0]:>8} {row[1]:>8}")

    # Specificity and sensitivity are the two halves balanced accuracy averages,
    # so printing them shows which side any drop lands on.
    print(f"  specificity (recall NORMAL)   : {cm[0, 0] / cm[0].sum():.4f}")
    print(f"  sensitivity (recall PNEUMONIA): {cm[1, 1] / cm[1].sum():.4f}")
    return bal_acc


if __name__ == "__main__":
    if MODEL not in FINAL_MODELS:
        raise ValueError(
            f"MODEL is {MODEL!r}; set it to one of {list(FINAL_MODELS)} in src/config.py"
        )

    checkpoint_name, model_class = FINAL_MODELS[MODEL]
    checkpoint_path = CKPT_DIR / checkpoint_name

    _, val_df, test_df = return_split()

    print(f"=== final evaluation: {MODEL} ===")

    y_true_val, y_pred_val, _ = predict(val_df, checkpoint_path, model_class=model_class)
    bal_acc_val = report("VALIDATION", y_true_val, y_pred_val)

    y_true_test, y_pred_test, _ = predict(
        test_df, checkpoint_path, model_class=model_class, verbose=False
    )
    bal_acc_test = report("TEST (single use)", y_true_test, y_pred_test)

    print(f"\nvalidation -> test: {bal_acc_val:.4f} -> {bal_acc_test:.4f}"
          f"  ({bal_acc_test - bal_acc_val:+.4f})")
