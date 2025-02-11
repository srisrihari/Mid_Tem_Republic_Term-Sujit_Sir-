# Thyroid Cancer Risk Prediction System

This project implements a machine learning system for thyroid cancer risk prediction using Multi-Layer Perceptron (MLP). It includes both a CLI application and a Flask web interface for making predictions.

## Project Components

1. **CLI Application**: Python-based command-line tool for batch predictions
2. **Flask Web Interface**: Web application for file uploads and real-time predictions
3. **MLP Model**: Deep learning model for thyroid cancer risk classification
4. **Docker Support**: Containerization for both CLI and web applications

## MLP Model Architecture

The model uses a Multi-Layer Perceptron with the following architecture:
- Input Layer: 16 features (preprocessed from raw data)
- Hidden Layer 1: 64 neurons with ReLU activation
- Hidden Layer 2: 32 neurons with ReLU activation
- Output Layer: 1 neuron with Sigmoid activation (binary classification)

### Model Features
- Binary Classification (0: Benign, 1: Malignant)
- PyTorch Implementation
- Configurable hyperparameters
- Model persistence (.pth format)

## Required Input Features

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Patient age (15-89 years) |
| Gender | Binary | Male/Female |
| Country | Categorical | Patient's country of origin |
| Ethnicity | Categorical | Patient's ethnic background |
| Family_History | Binary | History of thyroid-related issues |
| Radiation_Exposure | Binary | History of radiation exposure |
| Iodine_Deficiency | Binary | Presence of iodine deficiency |
| Smoking | Binary | Smoking status |
| Obesity | Binary | Obesity status |
| Diabetes | Binary | Diabetes status |
| TSH_Level | Numeric | Thyroid-Stimulating Hormone level (0.10-10.00) |
| T3_Level | Numeric | Triiodothyronine level (0.50-3.50) |
| T4_Level | Numeric | Thyroxine level (4.50-12.00) |
| Nodule_Size | Numeric | Size of thyroid nodule (0.00-5.00 cm) |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/srisrihari/Mid_Tem_Republic_Term-Sujit_Sir-.git
cd thyroid-cancer-prediction
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Applications

### CLI Application

1. Using Python:
```bash
python src/app.py --input data/test.csv --output predictions.csv --model models/model3_final.pth --verbose
```

2. Using Docker:
```bash
# Build the CLI image
docker build -f Dockerfile.cli -t thyroid-cli .

# Run predictions
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models thyroid-cli --input /app/data/test.csv --output /app/data/predictions.csv --model /app/models/model3_final.pth
```

### Web Application

1. Using Python:
```bash
python src/flask_app.py
```

2. Using Docker:
```bash
# Build the web app image
docker build -t thyroid-web .

# Run the web server
docker run -p 5000:5000 -v $(pwd)/models:/app/models thyroid-web
```

## Docker Implementation

The project includes two Dockerfiles:
1. `Dockerfile`: Web application container
2. `Dockerfile.cli`: CLI application container

Key features:
- Multi-stage builds for optimization
- Volume mounting for data persistence
- Environment variable configuration
- Non-root user execution
- Health checks

## Git Workflow

1. Main Branches:
   - `main`: Production-ready code


## Model Evaluation

The system includes comprehensive model evaluation:

1. Metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC

2. Visualization:
   - Confusion Matrix
   - ROC Curve
   - Feature Importance

3. Hyperparameter Tuning:
   - Learning Rate
   - Hidden Layer Sizes
   - Dropout Rates
   - Batch Size

## API Endpoints

### Web Interface
- `GET /`: Home page with upload form
- `POST /predict`: Process CSV file and return predictions
- `POST /upload_model`: Upload new model file
- `GET /health`: System health check

### CLI Commands
- `--input`: Input CSV file path
- `--output`: Output predictions path
- `--model`: Model file path
- `--verbose`: Enable detailed logging

## License

This project is licensed under the MIT License - see the LICENSE file for details. 