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
- **Normalise with mean 0.5700 / std 0.1791**, computed on the training split after the resize and
  crop, not on the raw files. Centre-cropping removes the dark borders and raises the mean from 
  0.4825 to 0.5700, while resizing smooths the image and lowers the standard deviation from 0.2383
  to 0.1791.
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

---

## Baselines

Splits after the exclusions above (283 reprocessed files, 32 exact duplicates found by content
hash). The 16 official validation images are marked `ignore` and left out.

| split | NORMAL | PNEUMONIA | total |
|---|---|---|---|
| train | 1,072 | 2,853 | 3,925 |
| val | 268 | 714 | 982 |
| test | 231 | 387 | 618 |

### Choosing the metric

Accuracy is useless here, but the usual replacement does not apply either: the rule of thumb says
to measure the minority class, and the minority class is `NORMAL`, but the clinically critical
one is `PNEUMONIA`, which is also the majority. Recall on pneumonia would therefore be maximised by
the degenerate model that predicts pneumonia for everything.

**Balanced accuracy**: the mean of sensitivity and specificity is used instead. The degenerate
classifier scores exactly 0.500 on every split under it, and unlike macro-F1 it does not depend on
class prevalence, so validation and test figures stay comparable despite their different balance.


### Results

| baseline | validation | test |
|---|---|---|
| trivial (always majority class) | 0.500 | 0.500 |
| logistic regression on metadata, unweighted | 0.791 | 0.756 |
| logistic regression on metadata, class-weighted | **0.860** | 0.801 |

Run log with confusion matrices in `experiments/runs.csv`.

**86% of the task, on validation, is solvable from acquisition metadata alone: image width, height
and aspect ratio. Without opening a single image.** That figure is the quantified version of the
resolution shortcut described above, and it is the context in which every later result has to be
read.

One thing it does not say. It is not a bar a CNN must clear to prove it learned anatomy: a CNN
never sees those numbers, since after resizing every image reaches the network with the same shape,
and the metadata survive only indirectly as resampling sharpness and geometric distortion.

---

## Baseline CNN

`src/model.py`, trained by `src/train.py` (hand-written loop, no high-level framework).

Four blocks of `Conv2d(3x3, padding=1) -> ReLU -> MaxPool(2)`, channels 1 → 16 → 32 → 64 → 128,
then global average pooling and a single `Linear(128, 2)`. Input 224x224, single channel.

**97,410 parameters, of which 99.7% are convolutional.** The classifier head contains only 258 
parameters. Global average pooling removes the spatial dimensions before classification, 
substantially reducing the number of parameters in the classification head compared with 
directly flattening the feature map.

Input size 224×224 was chosen because the pretrained backbones used for the final comparison 
expect that size, keeping the comparison consistent.

Training: batch size 32, Adam at 1e-3, `CrossEntropyLoss`, 30 epochs, seed 42. Grouped validation
split, evaluation on validation only.

### Result

**Best validation balanced accuracy: 0.9578 (epoch 28)**, against 0.860 for the metadata-only
baseline and 0.500 for the trivial one. Per-epoch history in `experiments/history_cnn_baseline_gap.csv`.

> This is the first result of the project, not its conclusion. The measurement protocol below
> revises how these figures should be compared, and the single evaluation on the held-out test set
> is at the end of this README.

### Diagnosis: the expected problem did not occur

|  | epoch 1 | epoch 10 | epoch 20 | epoch 30 |
|---|---|---|---|---|
| train loss | 0.503 | 0.180 | 0.135 | 0.092 |
| val loss | 0.364 | 0.162 | 0.121 | 0.098 |

The two losses fall together and stay within a few thousandths of each other to the end. There is no
overfitting: the validation loss never turns up, and both curves were still descending when training
stopped.

That was not the prediction. The standard heuristic, small dataset, no regularisation, thirty
epochs, expects memorisation. The absence of overfitting is therefore not evidence that the model lacks sufficient capacity to memorise the training set. Rather, the convolutional architecture imposes a strong inductive bias through local connectivity and weight sharing, while global average pooling substantially reduces the dimensionality of the representation passed to the classifier. These choices reduce the model's effective capacity compared with a conventional convolutional network followed by a large fully connected head, but they do not prevent memorisation in principle.

### Controlled comparison: the same network with a dense head

![Training curves: GAP against flatten head](figures/gap_vs_flatten_curves.png)

Swapping global average pooling for flatten + dense (97,410 to 3,308,802 parameters, same seed and
same pipeline) produces textbook overfitting: training loss reaches 0.0000 by epoch 22, validation
loss bottoms at epoch 8 and rises from there. Balanced accuracy does not follow, it stays around
0.97 throughout.

The extra capacity bought speed of fitting rather than generalisation: the large model is flat from
epoch 1, while the small one is still climbing at epoch 30.

---

## Measurement protocol

Extending the baseline to 80 epochs showed that the absence of overfitting at 30 was premature, not
real: validation loss bottoms at epoch 55 and rises afterwards, and the train/validation loss ratio
reaches 4x by epoch 80. Global average pooling delays memorisation by roughly fifty epochs, it does
not prevent it.

That run also exposed a measurement problem. The best epoch out of 80 draws beats the best out of 30
even with no real improvement, so the apparent gain was inflated. From there on:

- **fixed epoch budget**, and configurations are only compared at the same budget
- the reported score is the **mean of the last five epochs**, not the best epoch, which is a biased
  estimator
- the noise floor is measured, not assumed: the same configuration run under two torch seeds
  differs by **0.17 points** (0.9581 against 0.9564, runs 007 and 008)

Working rule for everything that follows: **differences below roughly half a point are not treated
as effects.** Reference figure for the from-scratch network at 80 epochs: **0.9640** (runs 006 and
013, spread 0.08 points).

Every run below is logged in `experiments/runs.csv` with its configuration, its confusion matrix
where relevant, and the reasoning behind its verdict.

---

## Improvements

Two interventions were tested against that reference. Neither improved the score. Both are reported.

### Cosine learning-rate decay: rejected (runs 009, 010)

Hypothesis: the 3-7 point swings between adjacent late epochs come from a learning rate that stays
at 1e-3 to the end, so annealing it should stabilise the estimate.

Late-epoch variance did drop, from std 0.0220 to 0.0032 on one seed. But that is guaranteed by
construction rather than evidence for the hypothesis: with `T_max` equal to the budget the learning
rate reaches zero and the weights stop moving, so the last epochs are near-copies of one another.

The seed replication is what settled it. The two annealed seeds land **1.65 points apart**, against
0.17 for the same two seeds without the schedule. Annealing freezes the model wherever its
trajectory happens to be, and with the learning rate at zero a run that is in a worse basin can no
longer leave it.

**The schedule does not reduce uncertainty about the model, it moves it out of sight**: from
variance inside one run to variance between runs, so a single annealed run looks more trustworthy
than it is. Rejected.

### Augmentation: kept, but not for the reason expected (runs 011-015)

Augmentations were chosen on anatomical grounds rather than by default. **`RandomHorizontalFlip` is
excluded**: the thorax is not symmetric.

What is kept corresponds to real acquisition variability: small rotation, translation and scale
(patient positioning, detector distance), mild brightness and contrast (exposure and dose).

A check on the augmented pipeline caught an artefact before any training time was spent. Mean and
standard deviation recomputed over 300 augmented images came out at **+28% on the standard
deviation**; inspecting a single image passed eight times through the chain showed why
`RandomAffine` fills the corners it rotates away with pure black, the most extreme value on the
scale, which after normalisation becomes -3.18, a value no real pixel reaches. Filling with the
dataset mean instead brought the statistics back to **+0.0003 / -0.0036**.

At the 40-epoch budget augmentation appeared to lose 1.42 points. It had not: in both seeds the best
epoch was epoch 40, the last one available, and both losses were still descending. When the maximum
falls on the edge of the budget, the budget is the constraint. Repeated at 80 epochs the deficit
fell to **0.38 points**, inside the spread between the augmented seeds. Extending the budget alone
removed most of the difference.

So augmentation buys no accuracy here. What it changes is the failure mode:

| | overfitting onset | train/val loss ratio at epoch 80 | final train loss |
|---|---|---|---|
| no augmentation | epochs 53 and 55 | 3.97x and 6.57x | 0.020-0.025 |
| geometric + photometric | epochs 73 and 76 | 2.57x and 2.43x | 0.076-0.079 |

No overlap between the two conditions on either seed. Augmentation roughly **doubles the time before
memorisation and halves its severity**. It is what makes a longer budget usable, not a gain at a
fixed one, and it is kept on that basis.

*Note on reading the curves: with augmentation, training loss ends above validation loss. That is
expected, not a bug. Training images are distorted and validation images are not, so the two losses
are no longer measured on the same distribution.*

---

## Transfer learning

ResNet18 with ImageNet weights, 224x224 input to match, the grayscale image repeated across three
channels rather than replacing `conv1` so the pretrained filters stay meaningful, and normalisation
switched to the ImageNet statistics the weights expect.

| | trainable parameters | epochs | validation |
|---|---|---|---|
| CNN from scratch | 97,410 | 80 | 0.9640 |
| ResNet18, backbone frozen | **1,026** | 20 | 0.9635 |
| ResNet18, fine-tuned | 11,177,538 | 20 | **0.9802** |

**A frozen ImageNet backbone matches the network trained from scratch**, to within five thousandths.
1,026 trainable parameters, no visual feature learned on this dataset at all. Generic natural-image
features already carry this task as far as training from scratch does; adapting them to radiographs
buys the further 1.67 points.

The two configurations are reported at different learning rates: 1e-4 for fine-tuning, 1e-3 for the
frozen head. Fine-tuning nudges pretrained weights and a high rate destroys
them; a randomly initialised linear head has to be learned from scratch and 1e-4 is too slow for it.
Run 017 is the frozen configuration at 1e-4, logged as discarded: it never converged, and it is kept
in the log because it is the reason the rates differ.

Fine-tuning also memorises the training set in a tenth of the epochs ( training loss 0.0003 by epoch
12 ) which means the mean-of-last-five protocol understates it: those epochs sit inside the
overfitted phase, while the useful zone was around 0.992. The protocol is kept unchanged for
comparability, but it is tuned to a network that converged far more slowly and it is not neutral
across architectures.

---

## Final evaluation on the test set

The official test split was used **once**, on two models selected on validation beforehand.

| | validation | test | specificity val → test | sensitivity val → test |
|---|---|---|---|---|
| CNN from scratch | 0.9706 | **0.7641** | 0.966 → **0.541** | 0.975 → 0.987 |
| ResNet18 fine-tuned | 0.9953 | **0.7766** | 0.996 → **0.558** | 0.994 → 0.995 |

**Both models lose roughly 21 points**, and the loss is entirely one-sided. Sensitivity does not
move, it rises slightly, with 5 and 2 missed pneumonias out of 387. Specificity halves: **106 and
102 of the 231 normal test radiographs are called PNEUMONIA.**

The two architectures differ by a factor of 115 in parameter count and by whether they were
pretrained on 1.2 million images, and they fail by the same amount in the same direction. So the
failure is not a property of either model.

The metadata baseline settles it. Without opening a single image, it shows the identical signature:
specificity 0.907 on validation against 0.693 on test, sensitivity rising from 0.814 to 0.910.
**Three models, one of which sees only width, height and aspect ratio, fail the same way.** When a
model that never looks at a pixel breaks like the two that do, the problem is not in the pixels.

### What actually happened

The validation set was drawn from the training pool. That was done correctly, stratified by class
and grouped by patient, but it means validation shares acquisition equipment, sites and period with
the training images. **It measured performance within a single acquisition regime, not
generalisation across regimes.**

The official test split is a separately collected set whose normal radiographs resemble, by
provenance, the pneumonia radiographs the models learned from. Both models over-call pneumonia on
them.

Every comparison in this project, GAP against a dense head, the learning-rate schedule,
augmentation, transfer learning, was therefore made inside that one regime. Moving outside it costs
21 points regardless of architecture. The architecture never mattered. The data did.

### One remedy that does not work

With sensitivity at 0.99 there is ample headroom to trade for specificity by lowering the decision
threshold. It cannot be done here: a threshold has to be tuned on validation, and **validation does
not exhibit the failure**, its specificity is 0.966. There is nothing to correct on a split that
does not show the defect, and tuning on the test split would turn it into a second validation set.

Fixing this needs data from a different acquisition source, not a different model or a different
threshold. It is the reason external validation is required in clinical machine learning, and this
is a textbook instance of it: see Zech et al., *PLOS Medicine* 2018, where pneumonia classifiers
learned to identify the hospital rather than the disease, and DeGrave et al., *Nature Machine
Intelligence* 2021, on the same failure in COVID-19 radiography.

---

## What this project shows

The headline number is not the point, but it is worth stating plainly: a network of **97,410
parameters trained from scratch** reaches within 1.6 points of an ImageNet backbone with 115 times
as many parameters, and matches that backbone exactly when its features are used frozen.

The result that matters is the other one. A model at 0.97 on validation drops to 0.76 on a
separately collected test set, and the log shows why, because a baseline that reads three numbers
from the image header, built before any network existed, breaks in exactly the same direction.
