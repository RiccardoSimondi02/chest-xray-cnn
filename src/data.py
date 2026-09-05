import pandas as pd

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
