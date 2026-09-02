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

(Note: zip extraction sometimes creates an extra nested folder, check before using the paths in the code.)

---

## Setup

```
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## Data inspection

Full analysis in `notebooks/01_exploration.ipynb`, inventory in `experiments/image_inventory.csv`.

| split | NORMAL | PNEUMONIA | share PNEUMONIA |
|---|---|---|---|
| train | 1,341 | 3,875 | 74.3% |
| val | 8 | 8 | 50.0% |
| test | 234 | 390 | 62.5% |

### What makes this dataset harder than it looks

- **The official validation set is unusable**: 16 images. It is rebuilt from the training set.
- **Train and test have different class balance** (74.3% vs 62.5% pneumonia), so the trivial
  classifier scores differently on each split and a model calibrated on train leans towards
  pneumonia at test time.
- **Image resolution alone predicts the class.** NORMAL median width 1,654 vs PNEUMONIA 1,160.
   This difference is so pronounced that it acts almost like a hidden label. Even if all images
   were resized to the same dimensions, the model could still detect the difference through 
   image sharpness or original aspect ratios, and learn to classify based on that rather 
   than the actual pathology.
- **283 images are an externally reprocessed batch**: stored as RGB, all `PNEUMONIA`, all in
  `train`, systematically smaller and more panoramic. The 76 smallest images in the dataset 
  are entirely contained in it.

### Decisions

- **Exclude the reprocessed batch**, rule `mode == "RGB"`, applied to all splits. Costs 7.3% of
  the training pneumonia images. *This removes the contaminated batch, not the resolution
  shortcut.* Raw data is never modified.
- **Convert everything to single channel** (correctness, not optimisation: all 283 RGB files are
  grayscale stored in an RGB container, verified channel-by-channel).
- **Normalise with mean 0.4825 / std 0.2383**, computed on the training split only, as the average
  of per-image means rather than pooled over pixels.
- **Preserve aspect ratio when resizing** (crop or pad, not a plain squeeze), because aspect ratio
  is class-correlated. Choice between crop and pad still open.
- **Validation split**: drawn from train only, stratified by class *and grouped by patient*, fixed
  seed, saved to disk as a manifest. Grouping is mandatory: 84% of training pneumonia images belong
  to patients that appear more than once.

### Experiment 

`NORMAL` filenames carry no patient identifier, so one was reconstructed from the filename
structure. Visual comparison was inconclusive, so a measurable consequence was tested instead: if
images sharing a key came from the same session, their dimensions should be more similar than
chance. Reshuffling the keys 200 times while preserving group sizes shows the observed value falls
inside the null distribution. So the key captures no coherence beyond chance.

The grouping is kept anyway, since merging two patients only costs flexibility while splitting one
produces leakage, but no claim is made that the validation split is patient-clean on `NORMAL`.