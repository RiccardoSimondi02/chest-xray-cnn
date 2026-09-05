from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

INVENTORY_CSV = "experiments/image_inventory.csv"
OUTPUT_CSV = "experiments/split.csv"
N_SPLITS = 5
VAL_FOLD = 0
RANDOM_STATE = 42

df = pd.read_csv(INVENTORY_CSV)
df = df[df["keep"]]

test_df = df[df["split"] == "test"]
val_df = df[df["split"] == "val"]
trainval_df = df[(df["split"] != "test") & (df["split"] != "val")].reset_index(drop=True)

y = trainval_df["label"].values
groups = trainval_df["patientid"].values

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
train_index, val_index = next(
    (splits for i, splits in enumerate(sgkf.split(trainval_df, y, groups)) if i == VAL_FOLD)
)

trainval_df = trainval_df.copy()
trainval_df.iloc[train_index, trainval_df.columns.get_loc("split")] = "train"
trainval_df.iloc[val_index, trainval_df.columns.get_loc("split")] = "val"

val_df = val_df.copy()
val_df["split"] = "ignore"

out_df = pd.concat([trainval_df, test_df, val_df])[["filepath", "split"]].reset_index(drop=True)
out_df.to_csv(OUTPUT_CSV, index=False)

print(out_df["split"].value_counts())
print(f"Saved to {OUTPUT_CSV}")
