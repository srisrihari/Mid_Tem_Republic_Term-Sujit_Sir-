import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import ADASYN
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
file_path = 'thyroid_cancer_risk_data.csv'
data = pd.read_csv(file_path)

# Separate binary and non-binary categorical columns
binary_cols = ['Gender', 'Family_History', 'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 'Obesity', 'Diabetes']
non_binary_cols = ['Country', 'Ethnicity']

# Label encoding for binary columns
label_encoders = {}
for column in binary_cols:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# One-hot encoding for non-binary categorical columns
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
transformed_cols = one_hot_encoder.fit_transform(data[non_binary_cols])

# Create a DataFrame with the one-hot encoded columns
one_hot_df = pd.DataFrame(transformed_cols, columns=one_hot_encoder.get_feature_names_out(non_binary_cols))

# Concatenate the one-hot encoded columns with the original data
data = pd.concat([data.drop(non_binary_cols, axis=1), one_hot_df], axis=1)

# Convert the target variable to numeric
data['Diagnosis'] = data['Diagnosis'].map({'Benign': 0, 'Malignant': 1})

# Create interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interaction_features = poly.fit_transform(data.drop(['Patient_ID', 'Thyroid_Cancer_Risk', 'Diagnosis'], axis=1))

# Create a DataFrame with the interaction features
interaction_df = pd.DataFrame(interaction_features, columns=poly.get_feature_names_out(data.drop(['Patient_ID', 'Thyroid_Cancer_Risk', 'Diagnosis'], axis=1).columns))

# Concatenate the interaction features with the original data
data = pd.concat([data, interaction_df], axis=1)

# Split the data into features and target
X = data.drop(['Patient_ID', 'Thyroid_Cancer_Risk', 'Diagnosis'], axis=1)
y = data['Diagnosis']

# Split the data into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance using ADASYN
adasyn = ADASYN(random_state=42)
X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

# Define the MLP model with dropout and L2 regularization
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5, activation=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, output_size))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Initialize the model, loss function, and optimizer
input_size = X_train_res.shape[1]
hidden_sizes = [128, 64, 32]  # Configurable hidden layers
output_size = 1
model = MLP(input_size, hidden_sizes, output_size, dropout_rate=0.5)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Lower learning rate and L2 regularization

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_res.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader for batch training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    y_pred_prob = test_outputs.round()
    accuracy = accuracy_score(y_test_tensor, y_pred_prob)
    precision = precision_score(y_test_tensor, y_pred_prob)
    recall = recall_score(y_test_tensor, y_pred_prob)
    f1 = f1_score(y_test_tensor, y_pred_prob)
    roc_auc = roc_auc_score(y_test_tensor, y_pred_prob)
    conf_matrix = confusion_matrix(y_test_tensor, y_pred_prob)
    print(f'Test Loss: {test_loss.item():.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1-Score: {f1:.4f}')
    print(f'Test ROC-AUC: {roc_auc:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

# Save the model
torch.save(model.state_dict(), 'mlp_model.pth')

# CLI Application
import argparse

def load_model(model_path):
    model = MLP(input_size, hidden_sizes, output_size, dropout_rate=0.5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, data):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(data_tensor)
        predictions = outputs.round().numpy()
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Thyroid Cancer Risk Prediction CLI')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output CSV file')
    parser.add_argument('--model', type=str, default='mlp_model.pth', help='Path to the trained model file')

    args = parser.parse_args()

    # Load the model
    model = load_model(args.model)

    # Load the input data
    input_data = pd.read_csv(args.input)
    input_data = scaler.transform(input_data)

    # Make predictions
    predictions = predict(model, input_data)

    # Save predictions to a CSV file
    output_df = pd.DataFrame(predictions, columns=['Prediction'])
    output_df.to_csv(args.output, index=False)
    print(f'Predictions saved to {args.output}')

if __name__ == '__main__':
    main()
