from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import io

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

# Configure upload folders
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
DATA_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv'}
ALLOWED_MODEL_EXTENSIONS = {'pth'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder='../templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_model_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_MODEL_EXTENSIONS

# Global variables for model and device
model = None
device = None

def initialize_model(model_path):
    """Initialize or reinitialize the model with a new model file."""
    global model, device
    try:
        model, device = load_model(model_path)
        return True
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        model, device = None, None
        return False

# Try to load the default model at startup
try:
    default_model_path = os.path.join(MODEL_FOLDER, 'model3_final.pth')
    if os.path.exists(default_model_path):
        initialize_model(default_model_path)
    else:
        logger.warning("No default model found at startup")
except Exception as e:
    logger.error(f"Error loading default model at startup: {str(e)}")

@app.route('/')
def home():
    """Home endpoint with web interface."""
    return render_template('index.html', model_loaded=model is not None)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    """Endpoint for uploading a new model file."""
    try:
        if 'model_file' not in request.files:
            return render_template('index.html', 
                                error='No model file uploaded',
                                model_loaded=model is not None)
        
        file = request.files['model_file']
        if file.filename == '':
            return render_template('index.html', 
                                error='No model file selected',
                                model_loaded=model is not None)
        
        if not allowed_model_file(file.filename):
            return render_template('index.html', 
                                error='Invalid file type. Only .pth files are allowed',
                                model_loaded=model is not None)
        
        # Save the model file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['MODEL_FOLDER'], filename)
        file.save(filepath)
        
        # Try to load the new model
        if initialize_model(filepath):
            return render_template('index.html', 
                                success='Model uploaded and loaded successfully',
                                model_loaded=True)
        else:
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template('index.html', 
                                error='Error loading the model. Please ensure it has the correct architecture',
                                model_loaded=model is not None)
    
    except Exception as e:
        logger.error(f"Error in upload_model endpoint: {str(e)}")
        return render_template('index.html', 
                            error=f'Error uploading model: {str(e)}',
                            model_loaded=model is not None)

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions on uploaded CSV data."""
    try:
        # Check if model is loaded
        if model is None:
            return render_template('index.html', 
                                error='No model loaded. Please upload a model first',
                                model_loaded=False)
        
        # Validate file upload
        if 'file' not in request.files:
            return render_template('index.html', 
                                error='No file uploaded',
                                model_loaded=True)
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', 
                                error='No file selected',
                                model_loaded=True)
        
        if not allowed_file(file.filename):
            return render_template('index.html', 
                                error='Invalid file type. Only CSV files are allowed',
                                model_loaded=True)
        
        # Save and process file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read and preprocess data
            data = pd.read_csv(filepath)
            processed_data = preprocess_data(data)
            
            # Ensure data is 2D
            if len(processed_data.shape) == 1:
                processed_data = processed_data.reshape(1, -1)
            
            # Convert to numpy array
            data_array = processed_data.to_numpy(dtype=np.float32)
            
            # Convert to tensor
            data_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)
            
            # Make predictions
            with torch.no_grad():
                outputs = model(data_tensor)
                predictions = (outputs.cpu().numpy() > 0.5).astype(int)
                probabilities = outputs.cpu().numpy()
            
            # Flatten predictions and probabilities
            predictions = predictions.flatten()
            probabilities = probabilities.flatten()
            
            # Prepare results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    'patient_id': data['Patient_ID'].iloc[i] if 'Patient_ID' in data.columns else i,
                    'prediction': int(pred),
                    'diagnosis': 'Malignant' if pred == 1 else 'Benign',
                    'probability': float(prob)
                }
                results.append(result)
            
            return render_template('index.html', 
                                predictions=results,
                                model_loaded=True)
        
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return render_template('index.html', 
                                error=f'Error processing file: {str(e)}',
                                model_loaded=True)
        
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return render_template('index.html', 
                            error=f'Server error: {str(e)}',
                            model_loaded=model is not None)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions (JSON response)."""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400
        
        # Save and process file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read and preprocess data
            data = pd.read_csv(filepath)
            processed_data = preprocess_data(data)
            
            # Ensure data is 2D
            if len(processed_data.shape) == 1:
                processed_data = processed_data.reshape(1, -1)
            
            # Convert to numpy array
            data_array = processed_data.to_numpy(dtype=np.float32)
            
            # Convert to tensor
            data_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)
            
            # Make predictions
            with torch.no_grad():
                outputs = model(data_tensor)
                predictions = (outputs.cpu().numpy() > 0.5).astype(int)
                probabilities = outputs.cpu().numpy()
            
            # Flatten predictions and probabilities
            predictions = predictions.flatten()
            probabilities = probabilities.flatten()
            
            # Prepare results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    'patient_id': data['Patient_ID'].iloc[i] if 'Patient_ID' in data.columns else i,
                    'prediction': int(pred),
                    'diagnosis': 'Malignant' if pred == 1 else 'Benign',
                    'probability': float(prob)
                }
                results.append(result)
            
            return jsonify({
                'success': True,
                'predictions': results,
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({
                'error': 'Error processing file',
                'details': str(e)
            }), 500
        
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)