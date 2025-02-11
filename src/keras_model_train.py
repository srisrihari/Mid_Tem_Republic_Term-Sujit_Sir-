# Importing required libs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from keras.models import Sequential
from keras.layers import Dense

# Reading the data
data = pd.read_csv('data/thyroid_cancer_risk_data.csv')

# EDA and Visualization
print("Data Shape:", data.shape)
print("\nData Info:")
data.info()
print("\nData Description:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Visualize target distribution
sns.countplot(x='Diagnosis', data=data)
plt.title('Distribution of Diagnosis')
plt.show()

# Categorizing columns
binarycol = ['Gender', 'Family_History', 'Radiation_Exposure', 'Iodine_Deficiency', 
             'Smoking', 'Obesity', 'Diabetes', 'Diagnosis']
nonbinarycol = ['Country', 'Ethnicity']

# Label encoding binary columns
label_encoder = {}
for col in binarycol:
    label_encoder[col] = LabelEncoder()
    data[col] = label_encoder[col].fit_transform(data[col])

# One hot encoding non-binary columns
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_data = one_hot_encoder.fit_transform(data[nonbinarycol])
one_hot_data = pd.DataFrame(encoded_data, 
                           columns=one_hot_encoder.get_feature_names_out(nonbinarycol))
data = pd.concat([data, one_hot_data], axis=1)
data.drop(nonbinarycol, axis=1, inplace=True)

# Save preprocessing objects
preprocessing_objects = {
    'label_encoders': label_encoder,
    'one_hot_encoder': one_hot_encoder,
    'binary_columns': binarycol,
    'nonbinary_columns': nonbinarycol
}

with open('models/preprocessing_objects.pkl', 'wb') as f:
    pickle.dump(preprocessing_objects, f)

# Splitting features and target
X = data.drop(['Patient_ID', 'Thyroid_Cancer_Risk', 'Diagnosis'], axis=1)
y = data['Diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Handle class imbalance using ADASYN
adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

print("After ADASYN - Training set shape:", X_train_resampled.shape)
print("Class distribution after ADASYN:")
print(pd.Series(y_train_resampled).value_counts())

# Building Keras Sequential Model
model = Sequential()
model.add(Dense(256, input_dim=X_train_resampled.shape[1], activation='relu', 
                kernel_initializer='normal'))
model.add(Dense(64, activation='relu', kernel_initializer='normal'))
model.add(Dense(32, activation='relu', kernel_initializer='normal'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))

# Model Summary
model.summary()

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train_resampled, y_train_resampled, 
                   batch_size=100, epochs=50, 
                   validation_split=0.2)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Make predictions
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Save the model
model.save('models/keras_model_final.h5')
print("\nModel saved as 'models/keras_model_final.h5'") 