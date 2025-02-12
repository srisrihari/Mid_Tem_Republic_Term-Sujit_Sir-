import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path
import os
import time

class MultiLayerMultiPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerMultiPerceptron, self).__init__()
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

def load_preprocessing_objects():
    """Load the preprocessing objects from pickle file."""
    try:
        with open('models/preprocessing_objects.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading preprocessing objects: {str(e)}")
        raise

def clean_data(data):
    """Clean input data before preprocessing."""
    data = data.copy()
    
    # Strip whitespace from string columns
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.strip()
    
    # Remove any hidden characters
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.replace('\r', '').str.replace('\n', '')
    
    return data

def validate_input_data(data):
    """Validate input data for required columns and data types."""
    required_columns = [
        'Age', 'Gender', 'Country', 'Ethnicity', 'Family_History',
        'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 'Obesity',
        'Diabetes', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size'
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Validate numeric columns
    numeric_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
    for col in numeric_cols:
        if not pd.to_numeric(data[col], errors='coerce').notna().all():
            raise ValueError(f"Column {col} contains non-numeric values")
    
    return True

def preprocess_data(data):
    """Preprocess input data for model prediction."""
    try:
        # Clean data first
        data = clean_data(data)
        
        # Load preprocessing objects
        prep_objects = load_preprocessing_objects()
        label_encoder = prep_objects['label_encoders']
        one_hot_encoder = prep_objects['one_hot_encoder']
        binary_cols = prep_objects['binary_columns']
        nonbinary_cols = prep_objects['nonbinary_columns']
        
        # Make a copy to avoid modifying the original data
        data = data.copy()
        
        # Validate input data
        validate_input_data(data)
        
        # Process binary columns first
        for col in binary_cols:
            if col in data.columns:
                # Convert to string and proper case based on column
                data[col] = data[col].astype(str).str.strip()
                if col == 'Gender':
                    # Capitalize first letter for Gender
                    data[col] = data[col].str.title()
                else:
                    # Capitalize for Yes/No columns
                    data[col] = data[col].str.capitalize()
                
                # Map values using label encoder
                data[col] = label_encoder[col].transform(data[col])
        
        # Process categorical columns
        if any(col in data.columns for col in nonbinary_cols):
            # Prepare categorical data
            cat_data = data[nonbinary_cols].copy()
            
            # Get categories from encoder
            encoder_categories = {
                col: one_hot_encoder.categories_[i].tolist() 
                for i, col in enumerate(nonbinary_cols)
            }
            
            # Map unknown categories to known ones
            for col in nonbinary_cols:
                cat_data[col] = cat_data[col].str.lower().str.strip()
                # Map any value not in known categories to 'other'
                cat_data[col] = cat_data[col].apply(
                    lambda x: x if x.lower() in [c.lower() for c in encoder_categories[col]] else 'other'
                )
                # Match the exact case from training
                for category in encoder_categories[col]:
                    mask = cat_data[col].str.lower() == category.lower()
                    cat_data.loc[mask, col] = category
            
            # Apply one-hot encoding
            encoded_data = one_hot_encoder.transform(cat_data)
            encoded_df = pd.DataFrame(
                encoded_data,
                columns=one_hot_encoder.get_feature_names_out(nonbinary_cols)
            )
            data = pd.concat([data, encoded_df], axis=1)
            data.drop(nonbinary_cols, axis=1, inplace=True)
        
        # Process numeric columns
        numeric_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].astype(np.float32)
        
        # Drop unnecessary columns
        drop_cols = ['Patient_ID', 'Label', 'Thyroid_Cancer_Risk', 'Diagnosis']
        data = data.drop(columns=[col for col in drop_cols if col in data.columns])
        
        return data
    
    except Exception as e:
        print(f"Error in preprocessing data: {str(e)}")
        raise

def load_model(model_path='models/model3_final.pth'):
    """Load the PyTorch model from the specified path."""
    try:
        print("⏳ Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiLayerMultiPerceptron()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully using {device}")
        return model, device
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

def get_valid_input(prompt, valid_options=None, is_numeric=False, example=""):
    while True:
        value = input(f"{prompt} (Example: {example}): ").strip()
        if is_numeric:
            try:
                return float(value)
            except ValueError:
                print("⚠️ Invalid input! Please enter a numeric value.")
        elif valid_options:
            if value in valid_options:
                return value
            print(f"⚠️ Invalid input! Please enter one of: {', '.join(valid_options)}")
        else:
            return value

def predict_single(model, device):
    """Make prediction for a single patient."""
    print("\n🔍 Enter patient details for thyroid cancer risk prediction:")
    
    features = {
        'Age': (True, "45"),
        'Gender': (False, "Male/Female"),
        'Country': (False, "India"),
        'Ethnicity': (False, "Asian"),
        'Family_History': (False, "Yes/No"),
        'Radiation_Exposure': (False, "Yes/No"),
        'Iodine_Deficiency': (False, "Yes/No"),
        'Smoking': (False, "Yes/No"),
        'Obesity': (False, "Yes/No"),
        'Diabetes': (False, "Yes/No"),
        'TSH_Level': (True, "2.5"),
        'T3_Level': (True, "1.2"),
        'T4_Level': (True, "8.5"),
        'Nodule_Size': (True, "2.3")
    }
    
    user_input = {}
    for feature, (is_numeric, example) in features.items():
        if feature == 'Gender':
            user_input[feature] = get_valid_input(feature, valid_options=['Male', 'Female'], example=example)
        elif feature in ['Family_History', 'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 'Obesity', 'Diabetes']:
            user_input[feature] = get_valid_input(feature, valid_options=['Yes', 'No'], example=example)
        else:
            user_input[feature] = get_valid_input(feature, is_numeric=is_numeric, example=example)
    
    df = pd.DataFrame([user_input])
    processed_data = preprocess_data(df)
    data_tensor = torch.tensor(processed_data.values, dtype=torch.float32).to(device)
    
    print("\n🧠 Running prediction...")
    time.sleep(1)
    
    with torch.no_grad():
        output = model(data_tensor)
        probability = output.cpu().numpy()[0][0]
        prediction = int(probability >= 0.5)
    
    print("\n🎯 Prediction Result:")
    print(f"{'❌ Low Risk' if prediction == 0 else '⚠️ High Risk'} of Thyroid Cancer")
    print(f"Probability: {probability:.2%}")

def predict_from_csv(model, device, input_path=None, output_path=None):
    """Make predictions from a CSV file."""
    if input_path is None:
        input_path = input("\n📂 Enter the CSV file path: ").strip()
    
    if not os.path.exists(input_path):
        print("❌ File not found! Please check the path and try again.")
        return
    
    try:
        print("\n📊 Processing data...")
        input_data = pd.read_csv(input_path)
        processed_data = preprocess_data(input_data)
        data_tensor = torch.tensor(processed_data.values, dtype=torch.float32).to(device)
        
        print("🧠 Running predictions...")
        with torch.no_grad():
            outputs = model(data_tensor)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int)
            probabilities = outputs.cpu().numpy()
        
        results = pd.DataFrame({
            'Patient_ID': input_data['Patient_ID'] if 'Patient_ID' in input_data.columns else range(len(predictions)),
            'Prediction': predictions.flatten(),
            'Risk_Level': ['High Risk' if p == 1 else 'Low Risk' for p in predictions.flatten()],
            'Probability': probabilities.flatten()
        })
        
        if output_path is None:
            os.makedirs("predictions", exist_ok=True)
            output_path = "predictions/thyroid_predictions.csv"
        
        results.to_csv(output_path, index=False)
        
        print(f"\n✅ Predictions saved as '{output_path}'!")
        print(f"📊 Summary:")
        print(f"Total patients: {len(predictions)}")
        print(f"High risk cases: {sum(predictions.flatten())}")
        print(f"Low risk cases: {len(predictions) - sum(predictions.flatten())}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Thyroid Cancer Risk Prediction CLI')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, help='Output predictions file path')
    parser.add_argument('--batch', action='store_true', help='Run in batch mode')
    args = parser.parse_args()

    try:
        model, device = load_model()
        
        # If both input and output are provided, run in batch mode
        if args.input and args.output:
            print(f"\n📊 Processing file: {args.input}")
            predict_from_csv(model, device, args.input, args.output)
        else:
            # Interactive mode
            while True:
                print("\n📌 Thyroid Cancer Risk Prediction CLI")
                print("1️⃣ Predict for a single patient")
                print("2️⃣ Predict from a CSV file")
                print("3️⃣ Exit")
                
                choice = input("\nEnter your choice: ").strip()
                
                if choice == '1':
                    predict_single(model, device)
                elif choice == '2':
                    predict_from_csv(model, device)
                elif choice == '3':
                    print("\n👋 Exiting... Have a great day!")
                    break
                else:
                    print("❌ Invalid choice! Please select 1, 2, or 3.")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())