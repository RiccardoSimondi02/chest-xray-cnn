import pandas as pd
import torch
from torchvision import transforms
from src.config import IMG_SIZE, MEAN, STD, LABELS_INT
from torch.utils.data import Dataset
from PIL import Image

def return_split():
    """
    Return the train, validation, and test splits.

    Returns:
    tuple: A tuple containing the train, validation, and test DataFrames.
    """
    INVENTORY_CSV = "experiments/image_inventory.csv"
    SPLIT_CSV = "experiments/split.csv"
    df_invetory = pd.read_csv(INVENTORY_CSV)
    df_split = pd.read_csv(SPLIT_CSV)

    df = pd.merge(df_invetory, df_split, on="filepath", how="left")
    df.drop(columns=["split_x"], inplace=True)
    df.rename(columns={"split_y": "split"}, inplace=True)
    df = df[df["keep"] == True]
    df_train, df_val, df_test = (
        df[df["split"] == "train"].reset_index(drop=True),
        df[df["split"] == "val"].reset_index(drop=True),
        df[df["split"] == "test"].reset_index(drop=True),
    )
    return df_train, df_val, df_test



def val_chain():
    return transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE), 
                        transforms.ToTensor(), transforms.Normalize(mean=[MEAN], std=[STD])])

def train_chain():
    return transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE), 
                        transforms.ToTensor(), transforms.Normalize(mean=[MEAN], std=[STD])])



class ChestXrayDataset(torch.utils.data.Dataset):
    def __init__(self, df, chain_func):
        self.df = df
        self.transform = chain_func()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"])
        img_transf = self.transform(img)
        label = LABELS_INT[row["label"]]       
        return img_transf, label