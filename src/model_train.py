# Importing required libs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import ADASYN
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import pickle

data = pd.read_csv('data/thyroid_cancer_risk_data.csv')  # reading the data from local file

## EDA and Pre-processing

data.head()

data.info()

data.describe()

data.isnull().sum()

# all the vizualization are visible on kaggle dataset main page so i didnt do it it again here but i have done one on diagnosis column as it is our target column

sns.countplot(x='Diagnosis',data=data)  #countplot of diagnosis
plt.show()

# categorizing the data into binary and non-binary columns
binarycol=['Gender','Family_History','Radiation_Exposure','Iodine_Deficiency','Smoking','Obesity','Diabetes','Diagnosis']
nonbinarycol =['Country', 'Ethnicity']

# Label encoding the binary columns
label_encoder ={}
for col in binarycol:
  label_encoder[col] = LabelEncoder()
  data[col] = label_encoder[col].fit_transform(data[col])

data.head()

# One hot encoding the non-binary columns
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_data = one_hot_encoder.fit_transform(data[nonbinarycol])
one_hot_data= pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(nonbinarycol))
data = pd.concat([data, one_hot_data], axis=1)
data.drop(nonbinarycol, axis=1, inplace=True)

data.head()

# After preprocessing steps, save the encoders and preprocessing objects
preprocessing_objects = {
    'label_encoders': label_encoder,
    'one_hot_encoder': one_hot_encoder,
    'binary_columns': binarycol,
    'nonbinary_columns': nonbinarycol
}

# Save preprocessing objects
with open('models/preprocessing_objects.pkl', 'wb') as f:
    pickle.dump(preprocessing_objects, f)

# Splitting the data into train and test
x = data.drop(['Patient_ID','Thyroid_Cancer_Risk','Diagnosis'], axis=1)
y = data['Diagnosis']
X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

# Saw Class Imblance in the Diagnosis column so balancing the data using ADASYN
ADASYN = ADASYN(random_state=42)
X_train_resampled,y_train_resampled= ADASYN.fit_resample(X_train,y_train)

pd.Series(y_train_resampled).value_counts()

X_train_resampled.shape[1]

# converting the data into tensors for pytorch model
X_train_tensor= torch.tensor(X_train_resampled.values,dtype=torch.float32)
X_test_tensor= torch.tensor(X_test.values,dtype=torch.float32)
y_train_tensor= torch.tensor(y_train_resampled.values,dtype=torch.float32)
y_test_tensor= torch.tensor(y_test.values,dtype=torch.float32)

## Single Layer Single Perceptron

# Creating class for the model
class SinglelayerSinglePerceptron(nn.Module):
    def __init__(self):
        super(SinglelayerSinglePerceptron, self).__init__()
        self.fc=nn.Linear(25, 1)
        self.sigmoid= nn.Sigmoid()

    def forward(self,x):
        out= self.fc(x)
        out= self.sigmoid(out)
        return out

model1= SinglelayerSinglePerceptron()  # creating object for the model

criterion= nn.MSELoss()  # defining the loss function
optimizer1= optim.Adam(model1.parameters(),lr=0.001) # defining the optimizer

num_epochs= 50 # number of epochs
batch_size= 100 # batch size

# defining the training function for all the models
def train_model(model, optimizer, criterion, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_epochs, batch_size):
  train_losses = []
  test_losses= []
  train_accuracies= []
  test_accuracies= []

  train_dataset= TensorDataset(X_train_tensor,y_train_tensor)
  train_loader= DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

  y_train_np = y_train_tensor.numpy().astype(int)
  y_test_np = y_test_tensor.numpy().astype(int)

  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_inputs.size(0)
    
    epoch_train_loss= running_loss/len(train_loader.dataset)
    

    model.eval()
    with torch.no_grad():
        # Training set evaluation
        train_outputs = model(X_train_tensor)
        train_loss = criterion(train_outputs, y_train_tensor.view(-1, 1))
        train_pred = (train_outputs.numpy() > 0.5).astype(int)
        train_accuracy = accuracy_score(y_train_np, train_pred)
        train_precision = precision_score(y_train_np, train_pred, zero_division=0)
        train_recall = recall_score(y_train_np, train_pred, zero_division=0)
        train_f1 = f1_score(y_train_np, train_pred, zero_division=0)
        
        # Test set evaluation
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor.view(-1, 1))
        test_pred = (test_outputs.numpy() > 0.5).astype(int)
        test_accuracy = accuracy_score(y_test_np, test_pred)
        test_precision = precision_score(y_test_np, test_pred, zero_division=0)
        test_recall = recall_score(y_test_np, test_pred, zero_division=0)
        test_f1 = f1_score(y_test_np, test_pred, zero_division=0)
        test_confusion = confusion_matrix(y_test_np, test_pred)
        test_auc = roc_auc_score(y_test_np, test_pred)
    
    train_losses.append(epoch_train_loss)
    test_losses.append(test_loss.item())
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Batch Train Loss: {epoch_train_loss:.4f},'
          f' Train Loss: {train_loss.item():.4f}, Train Accuracy: {train_accuracy:.4f},'
          f' Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, '
          f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}, '
          f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, '
          f'Test ROC-AUC: {test_auc:.4f}')

  plt.figure()
  plt.plot(train_accuracies, label='Train Accuracy')
  plt.plot(test_accuracies, label='Test Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()

  plt.figure()
  plt.plot(train_losses, label='Train Loss')
  plt.plot(test_losses, label='Test Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

# training the model
train_model(model1,optimizer1,criterion,X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,num_epochs,batch_size)

# Predictions function for all the models
def get_predictions(model, X_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (outputs.numpy() > 0.5).astype(int)
    return predictions

preds_model1 = get_predictions(model1, X_test_tensor) # getting the predictions for the model
print("Model 1 Test Predictions (first 10):", preds_model1[:10])

torch.save(model1.state_dict(), "models/model1_final.pth")  #saving the model for future use



## Single Layer Multi-Perceptron

# Creating class for the model
class SinglelayerMultiPerceptron(nn.Module):
  def __init__(self):
    super(SinglelayerMultiPerceptron,self).__init__()
    self.fc1= nn.Linear(25,64)
    self.relu= nn.ReLU()
    self.fc2= nn.Linear(64,1)
    self.sigmoid= nn.Sigmoid()
    
  def forward(self,x):
    out= self.fc1(x)
    out= self.relu(out)
    out= self.fc2(out)
    out= self.sigmoid(out)
    return out

model2= SinglelayerMultiPerceptron()  # creating object for the model2

criterion= nn.MSELoss() # defining the loss function
optimizer2= optim.Adam(model2.parameters(),lr=0.001) # defining the optimizer

num_epochs= 50 # number of epochs
batch_size= 50 # batch size

train_model(model2,optimizer2,criterion,X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,num_epochs,batch_size)  # training the model

preds_model2 = get_predictions(model2, X_test_tensor) # getting the predictions for the model
print("Model 2 Test Predictions (first 10):", preds_model2[:10])

torch.save(model2.state_dict(), "models/model2_final.pth") #saving the model for future use



## Multi Layer Multi Perceptron

# Creating class for the model
class MultilayerMultiPerceptron(nn.Module):
  def __init__(self):
    super(MultilayerMultiPerceptron,self).__init__()
    self.fc1= nn.Linear(25,64)
    self.relu1= nn.ReLU()
    self.fc2= nn.Linear(64,32)
    self.relu2= nn.ReLU()
    self.fc3= nn.Linear(32,1)
    self.sigmoid= nn.Sigmoid()



  def forward(self,x):
    out= self.fc1(x)
    out= self.relu1(out)
    out= self.fc2(out)
    out= self.relu2(out)
    out= self.fc3(out)
    out= self.sigmoid(out)
    return out

# creating object for the model3
model3= MultilayerMultiPerceptron()

criterion= nn.MSELoss()  # defining the loss function
optimizer3= optim.Adam(model3.parameters(),lr=0.001)  # defining the optimizer

num_epochs= 25 # number of epochs
batch_size= 50 # batch size

train_model(model3,optimizer3,criterion,X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,num_epochs,batch_size)

preds_model3 = get_predictions(model3, X_test_tensor)
print("Model 3 Test Predictions (first 10):", preds_model3[:10])

torch.save(model3.state_dict(), "models/model3_final.pth") #saving the model for further use
