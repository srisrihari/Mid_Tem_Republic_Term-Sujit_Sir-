#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ================================
# Define the MLP Model Architecture
# ================================
class MultilayerMultiPerceptron(nn.Module):
    def __init__(self):
        super(MultilayerMultiPerceptron, self).__init__()
        self.fc1 = nn.Linear(25, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

def load_model(model_path):
    """
    Loads the saved model state from the given .pth file.
    """
    model = MultilayerMultiPerceptron()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ================================
# Pre-processing Function
# ================================
def preprocess_data(df):
    """
    Applies the same pre-processing steps as in training:
      - Label encodes binary columns.
      - One-hot encodes non-binary columns.
      - Drops columns that are not features.
    
    Returns:
      - df_features: DataFrame with only the features (in the proper order).
      - df_full: Original DataFrame (possibly including a ground truth 'Diagnosis' column).
    """
    df_full = df.copy()  # keep a copy with all columns
    
    # Define the columns as used in training
    binary_cols = ['Gender', 'Family_History', 'Radiation_Exposure',
                   'Iodine_Deficiency', 'Smoking', 'Obesity', 'Diabetes']
    nonbinary_cols = ['Country', 'Ethnicity']
    
    # Label encode binary columns (if they exist in the data)
    for col in binary_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # One-hot encode non-binary columns (if present)
    if set(nonbinary_cols).issubset(df.columns):
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        ohe_features = ohe.fit_transform(df[nonbinary_cols])
        ohe_feature_names = ohe.get_feature_names_out(nonbinary_cols)
        df_ohe = pd.DataFrame(ohe_features, columns=ohe_feature_names, index=df.index)
        df = pd.concat([df, df_ohe], axis=1)
        df = df.drop(columns=nonbinary_cols)
    
    # Drop columns that were not used for training.
    # Note: We drop 'Patient_ID', 'Thyroid_Cancer_Risk', and 'Diagnosis' (if present)
    cols_to_drop = []
    for col in ['Patient_ID', 'Thyroid_Cancer_Risk', 'Diagnosis']:
        if col in df.columns:
            cols_to_drop.append(col)
    df_features = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df_features, df_full

# ================================
# Main function for the CLI app
# ================================
def main(args):
    # Load the CSV file provided by the user.
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Pre-process the data so it matches the training format.
    df_features, df_full = preprocess_data(df)
    
    # Check if the feature dimensionality is as expected (should be 25)
    if df_features.shape[1] != 25:
        print(f"Warning: Expected 25 feature columns but got {df_features.shape[1]}. Please verify your input file.")
    
    # Convert features to a torch tensor.
    X_tensor = torch.tensor(df_features.values, dtype=torch.float32)
    
    # Load the saved MLP model.
    model = load_model(args.model_path)
    
    # Get predictions (thresholding at 0.5)
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (outputs.numpy() > 0.5).astype(int).flatten()
    
    # Add predictions to the original dataframe.
    df_full['Prediction'] = predictions
    try:
        df_full.to_csv(args.output, index=False)
        print(f"Predictions saved successfully to {args.output}")
    except Exception as e:
        print(f"Error writing output file: {e}")
    
    # Display summary statistics.
    total_records = len(predictions)
    positive = np.sum(predictions)
    negative = total_records - positive
    print("\n--- Prediction Summary ---")
    print(f"Total records: {total_records}")
    print(f"Predicted Positive (e.g., backordered / malignant): {positive}")
    print(f"Predicted Negative (e.g., not backordered / benign): {negative}")
    
    # If ground truth exists (i.e. a 'Diagnosis' column is present), show evaluation metrics.
    if 'Diagnosis' in df_full.columns:
        # Assuming Diagnosis was label encoded similarly (0/1).
        y_true = df_full['Diagnosis'].values
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        print("\n--- Evaluation Metrics (Ground Truth Provided) ---")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

# ================================
# Argument Parsing
# ================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MLP Prediction CLI Application - Predict using a saved multilayer MLP model"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Path to the input CSV file containing features."
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Path where the CSV with predictions will be saved."
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='model3_final.pth',
        help="Path to the saved MLP model (.pth file)."
    )
    args = parser.parse_args()
    main(args)
