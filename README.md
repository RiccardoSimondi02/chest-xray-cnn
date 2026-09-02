# Chest X-Ray CNN: Pneumonia Classifier

Image classifier for chest X-rays, distinguishing `NORMAL` from `PNEUMONIA`.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Kaggle, dataset by Paul Mooney.

- ~5,800 pediatric chest X-rays, grayscale
- 2 classes: `NORMAL` / `PNEUMONIA`
- Already split into `train` / `val` / `test`

The dataset is **not included in the repository**. To reproduce it:

1. Download the zip from the dataset's Kaggle page (a Kaggle account is required).
2. Extract it into the `data/` folder, making sure the final structure looks like this:

```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

(Note: zip extraction sometimes creates an extra nested folder — check before using the paths in the code.)

---

## Setup

```
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```