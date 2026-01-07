# 🛰️ Satellite Imagery-Based Property Valuation

## 1. Project Overview

This project implements a **Multimodal Regression Pipeline** designed to predict residential property values by combining two complementary data modalities:

- **Tabular housing features** (e.g., bedrooms, square footage, location stats)
- **Satellite imagery** capturing environmental and neighborhood context

Traditional valuation models rely heavily on structured data and often fail to capture qualitative factors such as green cover, neighborhood density, or proximity to water. By integrating a **Convolutional Neural Network (CNN)** with a **Multi-Layer Perceptron (MLP)**, this system learns visual patterns from satellite imagery and fuses them with tabular features to improve prediction accuracy beyond standalone tabular models.

---

## 2. Repository Structure

Once you implement the project correctly, the folder structure should look like:

```text
├── data/
│   ├── images/                # Downloaded satellite images (id.jpg)
│   ├── train(1).xlsx          # Raw training dataset
│   ├── test2.xlsx             # Raw test dataset (final inference)
│   ├── train_cleaned.csv      # cleaned training data set
│   └── test_cleaned.csv       # Intermediate processed files
│
├── models/
│   ├── valuation_model.pth    # Trained PyTorch model weights
│   └── scaler.pkl             # Serialized StandardScaler
│
├── reports/
├   ├── model_test_results.ipynb   # Evaluation, inference & visualization
│   ├── submission.csv
│   └── training_curve.png
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   └── models.py
│
│── data_fetcher.py            # Satellite image downloader
├── preprocessing.ipynb        # Data cleaning & EDA
├── model_training.ipynb       # Multimodal training pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 3. Installation & Setup Guide

### Prerequisites

- Python 3.8 – 3.11  
  _(Avoid latest Python versions for CUDA compatibility)_
- Git
- CUDA-enabled GPU (recommended for CNN training)

---

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/MTHC582/Satellite_Imagery_Based_PropValuation
cd satellite-property-valuation
```

#### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Execution Order (How to Run)

### Step 1: Data Acquisition

- I have used the MapBox API, so go to the official web and grab a default token api.
- Now paste this API key inside a file named .env as,  
  MAPBOX_KEY = "your-api-key"
- Run the below mentioned .py file
- **Input:** `data/train(1).xlsx`
- **Output:** Images saved to `data/images/`

```bash
python src/data_fetcher.py
```

---

### Step 2: Preprocessing & Analysis

Open and run `preprocessing.ipynb`.

**Actions:**

- Data cleaning
- Handling missing values
- Feature engineering
- Exploratory Data Analysis (EDA)

**Goal:** Validate data integrity before training.

---

### Step 3: Model Training

Open and run `model_training.ipynb`.

**Actions:**

- Load satellite images and tabular features
- Apply log-transformation to target price
- Standardize numerical features
- Train ResNet-18 + MLP multimodal network

**Outputs:**

- `models/valuation_model.pth`
- `models/scaler.pkl`

---

### Step 4: Evaluation & Inference

Open and run `model_test_results.ipynb`.

**Actions:**

- Load trained model and scaler
- Run inference on validation set
- Convert predictions back from log-scale
- Generate Grad-CAM heatmaps

**Outputs:**

- RMSE metrics
- Visualization plots

---

## 5. Methodology & Architecture

### Data Pipeline

#### Tabular Data

- Normalized using `StandardScaler`
- Target variable (`price`) is log-transformed using `np.log1p`

#### Image Data

- Resized to `224 × 224`
- Normalized using ImageNet statistics:
  ```text
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]
  ```

---

### Multimodal Network (Late Fusion)

The architecture consists of two parallel branches:

#### Visual Branch (CNN)

- ResNet-18 backbone
- Produces a 512-dimensional image embedding

#### Tabular Branch (MLP)

- Fully connected neural network
- Processes 17 standardized numerical features

#### Fusion Layer

- Concatenates visual and tabular embeddings
- Final regression head predicts property value

---

## 6. Key Highlights

- Multimodal learning improves valuation accuracy
- Satellite imagery captures environmental context
- Log-scaled targets stabilize neural network training
- Grad-CAM provides visual explainability
