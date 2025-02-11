import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiLayerMultiPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerMultiPerceptron, self).__init__()
        self.fc1 = nn.Linear(16, 64)
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
        # Make a copy to avoid modifying the original data
        data = data.copy()
        
        # Validate input data
        validate_input_data(data)
        
        # Binary columns
        binary_cols = ['Gender', 'Family_History', 'Radiation_Exposure', 
                      'Iodine_Deficiency', 'Smoking', 'Obesity', 'Diabetes']
        
        # Non-binary columns for one-hot encoding
        nonbinary_cols = ['Country', 'Ethnicity']
        
        # Numeric columns
        numeric_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
        
        # Columns to drop
        drop_cols = ['Patient_ID', 'Label', 'Thyroid_Cancer_Risk', 'Diagnosis']
        
        # Process binary columns
        binary_mapping = {
            'Yes': 1, 'No': 0,
            'Male': 1, 'Female': 0,
            True: 1, False: 0,
            'true': 1, 'false': 0,
            'TRUE': 1, 'FALSE': 0,
            'yes': 1, 'no': 0,
            'male': 1, 'female': 0
        }
        
        # Convert binary columns
        for col in binary_cols:
            data[col] = data[col].astype(str).str.lower()
            data[col] = data[col].map(binary_mapping)
            if data[col].isna().any():
                raise ValueError(f"Invalid values in column {col}")
            data[col] = data[col].astype(np.float32)
        
        # Convert numeric columns
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            if data[col].isna().any():
                raise ValueError(f"Invalid numeric values in column {col}")
            data[col] = data[col].astype(np.float32)
        
        # Process categorical columns
        for col in nonbinary_cols:
            data[col] = data[col].astype(str).str.lower().str.strip()
        
        # Define expected categories for each categorical column
        expected_categories = {
            'Country': ['india', 'china', 'other'],
            'Ethnicity': ['caucasian', 'asian', 'other']
        }
        
        # Create dummy variables with expected categories
        dummy_dfs = []
        for col in nonbinary_cols:
            # Map any value not in expected categories to 'other'
            data[col] = data[col].apply(lambda x: x if x in expected_categories[col] else 'other')
            # Create dummies with all expected categories
            dummies = pd.get_dummies(data[col], prefix=col)
            # Add missing columns with zeros if they don't exist
            for cat in expected_categories[col]:
                col_name = f"{col}_{cat}"
                if col_name not in dummies.columns:
                    dummies[col_name] = 0
            # Drop the 'other' category column
            other_col = f"{col}_other"
            if other_col in dummies.columns:
                dummies = dummies.drop(columns=[other_col])
            # Sort columns to ensure consistent order
            dummies = dummies.reindex(sorted(dummies.columns), axis=1)
            dummy_dfs.append(dummies)
        
        # Drop unnecessary columns
        data = data.drop(columns=[col for col in drop_cols if col in data.columns])
        
        # Combine all features in a specific order
        feature_data = pd.concat([
            data[numeric_cols],  # 5 features
            data[binary_cols],   # 7 features
            dummy_dfs[0],        # 2 features for Country (india, china)
            dummy_dfs[1]         # 2 features for Ethnicity (asian, caucasian)
        ], axis=1)
        
        # Convert to float32
        feature_data = feature_data.astype(np.float32)
        
        # Ensure we have exactly 16 features
        expected_features = (
            len(numeric_cols) +  # 5 numeric features
            len(binary_cols) +   # 7 binary features
            (len(expected_categories['Country']) - 1) +  # 2 country features (excluding 'other')
            (len(expected_categories['Ethnicity']) - 1)  # 2 ethnicity features (excluding 'other')
        )
        
        if feature_data.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features but got {feature_data.shape[1]}")
        
        # Log feature information
        logger.debug(f"Preprocessed data shape: {feature_data.shape}")
        logger.debug(f"Preprocessed columns: {feature_data.columns.tolist()}")
        logger.debug(f"Feature data types: {feature_data.dtypes}")
        
        return feature_data
    
    except Exception as e:
        logger.error(f"Error in preprocessing data: {str(e)}")
        raise

def load_model(model_path):
    """Load the PyTorch model from the specified path."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiLayerMultiPerceptron()
        
        # Load state dict and handle potential key mismatches
        state_dict = torch.load(model_path, map_location=device)
        
        # If the model was saved with a different architecture, we need to reinitialize
        if 'fc1.weight' in state_dict and state_dict['fc1.weight'].shape[1] != 16:
            logger.info("Reinitializing model with correct input size")
            model = MultiLayerMultiPerceptron()  # Reinitialize with correct size
        else:
            model.load_state_dict(state_dict)
            
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully using device: {device}")
        return model, device
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict(model, data, device):
    """Make predictions using the loaded model."""
    try:
        # Ensure data is 2D
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
            
        # Convert to numpy array
        data_array = data.to_numpy(dtype=np.float32)
        
        # Convert to tensor
        data_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(data_tensor)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int)
            probabilities = outputs.cpu().numpy()
        
        return predictions.flatten(), probabilities.flatten()
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Thyroid Cancer Risk Prediction CLI')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output CSV file')
    parser.add_argument('--model', type=str, default='models/model3_final.pth', help='Path to the trained model file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    try:
        # Set logging level based on verbose flag
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        # Validate model file
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {args.model}")
        
        # Create output directory if it doesn't exist
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess input data
        logger.info(f'Loading data from {args.input}...')
        input_data = pd.read_csv(args.input)
        processed_data = preprocess_data(input_data)
        
        # Load the model
        logger.info(f'Loading model from {args.model}...')
        model, device = load_model(args.model)

        # Make predictions
        logger.info('Making predictions...')
        predictions, probabilities = predict(model, processed_data, device)

        # Prepare output
        results = pd.DataFrame({
            'Patient_ID': input_data['Patient_ID'] if 'Patient_ID' in input_data.columns else range(len(predictions)),
            'Prediction': predictions,
            'Risk_Level': ['High Risk' if p == 1 else 'Low Risk' for p in predictions],
            'Probability': probabilities
        })
        
        # Save predictions
        results.to_csv(output_path, index=False)
        logger.info(f'Predictions saved to {output_path}')
        logger.info(f'Total samples processed: {len(predictions)}')
        logger.info(f'High risk cases: {sum(predictions)}')
        logger.info(f'Low risk cases: {len(predictions) - sum(predictions)}')

    except Exception as e:
        logger.error(f'Error: {str(e)}')
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())