import pandas as pd
import torch
from torchvision import transforms
from src.config import AUGMENT, IMG_SIZE, MEAN, STD, LABELS_INT, MODEL
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



def _model_preprocessing(model_name):
    """
    Single source of truth for how a given MODEL wants its input preprocessed:
    (num_channels, mean, std). train_chain and val_chain both call this, so
    they can never diverge on channels/normalization for a given backbone.
    """
    if model_name in ("resnet18_finetune", "resnet18_frozen"):
        # pretrained on ImageNet: 3 channels, ImageNet stats
        return 3, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # our own CNN, trained from scratch on grayscale chest x-rays
    return 1, [MEAN], [STD]


def val_chain():
    channels, mean, std = _model_preprocessing(MODEL)
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=channels),
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def train_chain():
    channels, mean, std = _model_preprocessing(MODEL)
    fill_value = round(MEAN * 255)

    transforms_list = [
        transforms.Grayscale(num_output_channels=channels),
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
    ]

    if AUGMENT == "geom_photo":
        transforms_list.append(
            transforms.RandomAffine(
                degrees=7,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                fill=fill_value
            )
        )
        transforms_list.append(
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1
            )
        )

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transforms_list)



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