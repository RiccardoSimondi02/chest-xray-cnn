IMG_SIZE = 224
MEAN = 0.5700
STD = 0.1791
LABELS = ["NORMAL", "PNEUMONIA"]
LABELS_INT = {label: i for i, label in enumerate(LABELS)}
LABELS_INVERSE = {
    0: "NORMAL",
    1: "PNEUMONIA"
}

"""

N_EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
SEED = 42

# "none" / "cosine"
SCHEDULER = "none"
# "none" / "geom_photo"
AUGMENT = "geom_photo"

"""

# transfer learning settings

# "cnn" / "resnet18_finetune" / "resnet18_frozen"
MODEL = "resnet18_finetune"

N_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SCHEDULER = "none"
AUGMENT = "none"
NUM_WORKERS = 4
SEED = 42