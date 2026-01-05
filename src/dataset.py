import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

# Custom made classes-lib
"""
    Custom Dataset to handle paired Data:
    1| Numerical Features (from Excel)
    2| Satellite Image (from Folder)
"""


class SatelliteDataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None,
        feature_cols=None,
        target_col=None,
        is_test=False,
    ):
        """
        Args:
            dataframe (pd.DataFrame)       : The preprocessed DataFrame.
            image_dir (string)             : Directory with all the images.
            transform (callable, optional) : PyTorch transforms for the images.
            feature_cols (list)            : List of numerical column names to use.
            target_col (string)            : Name of the target column (price).
        """

        # Init to the object.
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

        # Default features to use if none are provided
        if feature_cols is None:
            self.feature_cols = [
                "bedrooms",
                "bathrooms",
                "sqft_living",
                "sqft_lot",
                "floors",
                "waterfront",
                "view",
                "condition",
                "grade",
                "sqft_above",
                "sqft_basement",
                "yr_built",
                "yr_renovated",
                "sqft_living15",
                "sqft_lot15",
                "lat",
                "long",
            ]
        else:
            self.feature_cols = feature_cols

        self.target_col = target_col

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 1_Get the Data Row
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx]
        id_val = str(row["id"])

        # 2_Load Image (Visual Input)
        img_name = os.path.join(self.image_dir, f"{id_val}.jpg")

        try:
            image = Image.open(img_name).convert("RGB")
        except (FileNotFoundError, OSError):
            # If image is missing by any kind or error or crAsh, we load it with a black image instead.
            # This also makes sure that the Trainin loop never crashes.
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        # 3_Load Numerical Features (Tabular Input)
        # We perform fillna(0) here (just in case), though preprocessing is made for that too!..
        features_data = row[self.feature_cols].fillna(0).values.astype(np.float32)
        features = torch.tensor(features_data, dtype=torch.float32)

        # 4_Load Target (Price) - Only if NOT in test mode
        if not self.is_test:
            target = torch.tensor(
                row["price"], dtype=torch.float32
            )  # Although defaults to float32.
            # Since GPU works eff in float32, hence needed to make sure/
            return image, features, target

        # For prediction (no target available)
        return image, features


"""
 SANITY CHECK
 Unit Test to make sure it works on the first row 
 To have a proper usage over trainin without any kind of errors
"""

if __name__ == "__main__":
    # TO check only when ran here.

    print("Testing Dataset Class...")

    # Load the actual file to test.
    try:
        df = pd.read_excel("data/train(1).xlsx")

        # Define a simple transform.
        simple_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

        # Initialize Dataset
        ds = SatelliteDataset(
            dataframe=df, image_dir="data/images", transform=simple_transform
        )

        print(f"Dataset Initialized. Found {len(ds)} rows.")

        # Grab Item 0 aka the firstt oneE.
        img, feats, price = ds[0]
        print(f"Image Shape: {img.shape} (Should be 3, 224, 224)")
        print(f"Features: {feats.shape} (Should be 17)")  # 15 original + lat/long
        print(f"Price: ${price.item():,.2f}")
        print("\nSUCCESS: The Dataset code is working!")

    except FileNotFoundError:  # Making sure the paths n file names match accordingly.
        print(
            "Could not find 'data/train(1).xlsx'. Make sure you are in the root folder."
        )
