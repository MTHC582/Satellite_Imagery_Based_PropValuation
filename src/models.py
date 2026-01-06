"""
Using ResNet18 (modern compared to ResNet50) from Microsofts pretrained models for feature extraction.
It is the industry standard for "Transfer Learning", On mid-sized datasets, It strikes
the perfect balance of depth and speed.
It is a lightweight yet powerful convolutional neural network that excels at image recognition.
Perfect for detecting houses and other structures with high accuracy and efficiency.
This is less prone to overfitting on noice.
"""

import torch
import torch.nn as nn
from torchvision import models


class ValuationModel(nn.Module):
    def __init__(self, num_tabular_features=17, dropout_rate=0.3):
        """
        Hybrid Model: Fuses Satellite Imagery (ResNet18) + Tabular Data (FFN).
        Args:
            num_tabular_features (int)  : Must match dataset columns (17: 15 stats + lat + long).
            dropout_rate (float)        : Probability of dropping neurons to prevent overfitting.
        """
        super().__init__()  # Calls the parent class nn.module and it initializes the nn.Module properly.

        # The EYES (Image Analysis)
        self.cnn = models.resnet18(weights="IMAGENET1K_V1")

        # ResNet18 usually outputs 1000 classes. We r goin to restrict that much
        # We replace the final classification layer with an Identity mapping
        # so we get the raw 512 internal visual features.
        self.cnn.fc = nn.Identity()

        # The CALCULATOR (Tabular Analysis)
        # A Feed-Forward Network (FFN) to process the Excel data.
        self.tabular_ffn = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),  # Normalizes data batch-by-batch for stability
            nn.Linear(64, 32),  # Compress down the 64 to 32...
            nn.ReLU(),
        )

        # Stage of combining Brain + Calculator
        # Visual Features (512) + Tabular Features (32) = 544 Total Inputs
        self.fusion_head = nn.Sequential(
            nn.Linear(512 + 32, 128),  # Compress down to 128
            nn.ReLU(),
            nn.Dropout(
                dropout_rate
            ),  # Crucial: Randomly sleeps neurons to prevent memorization
            nn.Linear(128, 64),  # COmpress to 64
            nn.ReLU(),
            nn.Linear(64, 1),  # Final Output: The Predicted Price (1 Single number)
        )

    def forward(self, image, tabular_data):
        # 1_Process Image
        x_image = self.cnn(image)  # Shape: [Batch, 512]

        # 2_Process Numbers
        x_tab = self.tabular_ffn(tabular_data)  # Shape: [Batch, 32]

        # 3_Fuse (Concatenate)
        # We brng the vectors side-by-side
        combined = torch.cat((x_image, x_tab), dim=1)  # Shape: [Batch, 544]

        # 4_Predict Price
        price = self.fusion_head(combined)
        return price


# SANITY CHECK / UNIT-TEST
if __name__ == "__main__":
    # TO check within the file itself.
    print("Testing ValuationModel Architecture...")

    # 1_Create a Dummy Image (Batch Size 2, 3 Channels, 224x224)
    # BAtch size as 2, since we r checkin 2 houses simult.!
    # Since RGB = 3 channels, where as 224 is a standard size for ResNet
    dummy_image = torch.randn(2, 3, 224, 224)

    # 2_Create Dummy Data (Batch Size 2, 17 Features)
    # 17 = 15 house stats + lat + long
    dummy_data = torch.randn(2, 17)

    # 3_Initialize Model
    try:
        model = ValuationModel(num_tabular_features=17)
        print("Model initialized successfully.")

        # 4_Pass Data through Model
        output = model(dummy_image, dummy_data)

        print(f"Input Image Shape: {dummy_image.shape}")
        print(f"Input Data Shape:  {dummy_data.shape}")
        print(f"Output Shape:      {output.shape} (Should be [2, 1])")
        print(f"Sample Prediction: ${output[0].item():,.2f}")

        print("\nSUCCESS: The Model is built and processing data!")
    except Exception as ex:
        print(f"\nERROR: {ex}")
