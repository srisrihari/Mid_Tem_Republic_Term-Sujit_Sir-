from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import io
import pickle

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
        
        # Log initial state
        print("Initial data columns: %s", data.columns.tolist())
        print("Binary columns from training: %s", binary_cols)
        print("Non-binary columns from training: %s", nonbinary_cols)
        
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
                
                print(f"Unique values in {col} before mapping: {data[col].unique()}")
                print(f"Expected classes for {col}: {label_encoder[col].classes_}")
                
                # Map values using label encoder directly
                try:
                    data[col] = label_encoder[col].transform(data[col])
                except ValueError as e:
                    print(f"Error encoding {col}: {str(e)}")
                    print(f"Unique values in {col}: {data[col].unique()}")
                    print(f"Label encoder classes: {label_encoder[col].classes_}")
                    raise
                
                print(f"Unique values in {col} after encoding: {data[col].unique()}")
        
        # Process categorical columns
        if any(col in data.columns for col in nonbinary_cols):
            # Prepare categorical data
            cat_data = data[nonbinary_cols].copy()
            
            # Get the actual categories from the encoder
            encoder_categories = {
                col: one_hot_encoder.categories_[i].tolist() 
                for i, col in enumerate(nonbinary_cols)
            }
            print(f"Encoder categories: {encoder_categories}")
            
            # Map unknown categories to 'other'
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
                
                print(f"Unique values in {col} after mapping: {cat_data[col].unique()}")
            
            # Apply one-hot encoding
            try:
                encoded_data = one_hot_encoder.transform(cat_data)
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=one_hot_encoder.get_feature_names_out(nonbinary_cols)
                )
                print(f"Encoded columns: {encoded_df.columns.tolist()}")
                data = pd.concat([data, encoded_df], axis=1)
                data.drop(nonbinary_cols, axis=1, inplace=True)
            except Exception as e:
                print(f"Error in one-hot encoding: {str(e)}")
                print(f"Categories in data: {[cat_data[col].unique() for col in nonbinary_cols]}")
                print(f"Expected categories: {encoder_categories}")
                raise
        
        # Process numeric columns
        numeric_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            if data[col].isna().any():
                raise ValueError(f"Invalid numeric values in column {col}")
            data[col] = data[col].astype(np.float32)
        
        # Drop unnecessary columns
        drop_cols = ['Patient_ID', 'Label', 'Thyroid_Cancer_Risk', 'Diagnosis']
        data = data.drop(columns=[col for col in drop_cols if col in data.columns])
        
        # Log final state
        print("Final data shape: %s", data.shape)
        print("Final columns: %s", data.columns.tolist())
        
        return data
    
    except Exception as e:
        print(f"Error in preprocessing data: {str(e)}")
        raise

def load_model(model_path):
    """Load the PyTorch model from the specified path."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiLayerMultiPerceptron()
        
        # Load state dict and handle potential key mismatches
        state_dict = torch.load(model_path, map_location=device)
        
        # If the model was saved with a different architecture, we need to reinitialize
        if 'fc1.weight' in state_dict and state_dict['fc1.weight'].shape[1] != 25:
            print("Reinitializing model with correct input size")
            model = MultiLayerMultiPerceptron()  # Reinitialize with correct size
        else:
            model.load_state_dict(state_dict)
            
        model.to(device)
        model.eval()
        print(f"Model loaded successfully using device: {device}")
        return model, device
    except Exception as e:
        print(f"Error loading model: {str(e)}")
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
        # Try to load preprocessing objects first
        load_preprocessing_objects()
        
        # Then load the model
        model, device = load_model(model_path)
        return True
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        model, device = None, None
        return False

# Try to load the default model at startup
try:
    default_model_path = os.path.join(MODEL_FOLDER, 'model3_final.pth')
    if os.path.exists(default_model_path):
        initialize_model(default_model_path)
    else:
        print("No default model found at startup")
except Exception as e:
    print(f"Error loading default model at startup: {str(e)}")

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
        print(f"Error in upload_model endpoint: {str(e)}")
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
        if model is None:
            return render_template('index.html', 
                                error='⚠️ No model loaded. Please upload a model first',
                                model_loaded=False)
        
        if 'file' not in request.files:
            return render_template('index.html', 
                                error='⚠️ No file uploaded',
                                model_loaded=True)
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', 
                                error='⚠️ No file selected',
                                model_loaded=True)
        
        if not allowed_file(file.filename):
            return render_template('index.html', 
                                error='⚠️ Invalid file type. Only CSV files are allowed',
                                model_loaded=True)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            print("📊 Processing uploaded file...")
            data = pd.read_csv(filepath)
            
            # Generate data summary
            summary = {
                'total_samples': len(data),
                'columns': data.columns.tolist(),
                'numeric_summary': {},
                'categorical_summary': {}
            }
            
            # Numeric columns summary
            numeric_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
            for col in numeric_cols:
                if col in data.columns:
                    summary['numeric_summary'][col] = {
                        'mean': float(data[col].mean()),
                        'median': float(data[col].median()),
                        'std': float(data[col].std()),
                        'min': float(data[col].min()),
                        'max': float(data[col].max())
                    }
            
            # Categorical columns summary
            categorical_cols = ['Gender', 'Country', 'Ethnicity', 'Family_History', 
                              'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 
                              'Obesity', 'Diabetes']
            for col in categorical_cols:
                if col in data.columns:
                    value_counts = data[col].value_counts()
                    summary['categorical_summary'][col] = {
                        str(k): int(v) for k, v in value_counts.items()
                    }
            
            # Preprocess data and make predictions
            processed_data = preprocess_data(data)
            data_tensor = torch.tensor(processed_data.values, dtype=torch.float32).to(device)
            
            print("🧠 Running predictions...")
            with torch.no_grad():
                outputs = model(data_tensor)
                predictions = (outputs.cpu().numpy() > 0.5).astype(int)
                probabilities = outputs.cpu().numpy()
            
            # Prepare prediction results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions.flatten(), probabilities.flatten())):
                result = {
                    'patient_id': int(data['Patient_ID'].iloc[i]) if 'Patient_ID' in data.columns else i,
                    'prediction': int(pred),
                    'diagnosis': 'Malignant' if pred == 1 else 'Benign',
                    'probability': float(prob)
                }
                results.append(result)
            
            # Calculate prediction statistics
            prediction_stats = {
                'total_predictions': len(predictions),
                'malignant_count': int(np.sum(predictions)),
                'benign_count': int(len(predictions) - np.sum(predictions)),
                'malignant_percentage': float(np.sum(predictions) / len(predictions) * 100),
                'average_probability': float(np.mean(probabilities))
            }
            
            return render_template('index.html', 
                                predictions=results,
                                data_summary=summary,
                                prediction_stats=prediction_stats,
                                success='✅ Predictions completed successfully!',
                                model_loaded=True)
        
        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")
            return render_template('index.html', 
                                error=f'❌ Error processing file: {str(e)}',
                                model_loaded=True)
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return render_template('index.html', 
                            error=f'❌ Server error: {str(e)}',
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
            print(f"Error processing file: {str(e)}")
            return jsonify({
                'error': 'Error processing file',
                'details': str(e)
            }), 500
        
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    # Initialize model
    default_model_path = os.path.join(app.config['MODEL_FOLDER'], 'model3_final.pth')
    if os.path.exists(default_model_path):
        initialize_model(default_model_path)
    
    # Run the app on 0.0.0.0 to make it accessible from outside the container
    app.run(host='0.0.0.0', port=5001)