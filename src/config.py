IMG_SIZE = 224
MEAN = 0.5700
STD = 0.1791
LABELS = ["NORMAL", "PNEUMONIA"]
LABELS_INT = {label: i for i, label in enumerate(LABELS)}
LABELS_INVERSE = {
    0: "NORMAL",
    1: "PNEUMONIA"
}

N_EPOCHS = 30
