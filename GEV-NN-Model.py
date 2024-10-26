# GEV-NN Model for Rare Event Classification

# Install necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, roc_curve, 
                             brier_score_loss, confusion_matrix, ConfusionMatrixDisplay, auc)
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
train_data = pd.read_csv("training_dataset.csv")
validation_data = pd.read_csv("validation_dataset.csv")

# Define target and features
target_column = 'event_indicator'  # Adjust as per your target variable
features = train_data.drop(['record_id', target_column], axis=1)
target = train_data[target_column]

val_features = validation_data.drop(['record_id', target_column], axis=1)
val_target = validation_data[target_column]

# Standardize features
scaler = StandardScaler()
trainX = scaler.fit_transform(features)
valX = scaler.transform(val_features)

# Convert to tensors
trainX = torch.tensor(trainX, dtype=torch.float32)
valX = torch.tensor(valX, dtype=torch.float32)
trainY = torch.tensor(target.values, dtype=torch.float32).view(-1, 1)
valY = torch.tensor(val_target.values, dtype=torch.float32).view(-1, 1)

# Create datasets and dataloaders
train_dataset = TensorDataset(trainX, trainY)
val_dataset = TensorDataset(valX, valY)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Training data shape after preprocessing:", trainX.shape)
print("Validation data shape after preprocessing:", valX.shape)

# Define the GEV activation function
class GevActivation(nn.Module):
    def forward(self, x):
        return torch.exp(-torch.exp(-x))

# Define the Autoencoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoder_layers, decoder_layers, dropout_rate=0.5):
        super(AutoEncoder, self).__init__()
        encoder = []
        decoder = []

        # Build the encoder
        for i in range(len(encoder_layers)):
            if i == 0:
                encoder.append(nn.Linear(input_dim, encoder_layers[i]))
            else:
                encoder.append(nn.Linear(encoder_layers[i-1], encoder_layers[i]))
            encoder.append(nn.ReLU())
            encoder.append(nn.Dropout(dropout_rate))
            encoder.append(nn.BatchNorm1d(encoder_layers[i]))

        # Build the decoder
        for i in range(len(decoder_layers)):
            if i == 0:
                decoder.append(nn.Linear(encoder_layers[-1], decoder_layers[i]))
            else:
                decoder.append(nn.Linear(decoder_layers[i-1], decoder_layers[i]))
            decoder.append(nn.ReLU())
            decoder.append(nn.Dropout(dropout_rate))
            decoder.append(nn.BatchNorm1d(decoder_layers[i]))

        decoder.append(nn.Linear(decoder_layers[-1], input_dim))
        decoder.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define the main GEV-NN model
class GEV_NN(nn.Module):
    def __init__(self, input_dim, encoder_layers, decoder_layers, sofnn_layers, mlp_neurons, dropout_rate=0.5):
        super(GEV_NN, self).__init__()

        self.autoencoder = AutoEncoder(input_dim, encoder_layers, decoder_layers, dropout_rate)

        # Build the SOFNN
        sofnn = []
        for i in range(len(sofnn_layers)):
            if i == 0:
                sofnn.append(nn.Linear(input_dim, sofnn_layers[i]))
            else:
                sofnn.append(nn.Linear(sofnn_layers[i-1], sofnn_layers[i]))
            sofnn.append(nn.Tanh())
            sofnn.append(nn.Dropout(dropout_rate))
            if sofnn_layers[i] > 1:
                sofnn.append(nn.BatchNorm1d(sofnn_layers[i]))
        sofnn.append(nn.Linear(sofnn_layers[-1], input_dim))
        sofnn.append(nn.Softmax(dim=1))
        self.sofnn = nn.Sequential(*sofnn)

        # Build the MLP for final prediction
        mlp_input_dim = input_dim + 2 + encoder_layers[-1]
        mlp = []
        for i in range(len(mlp_neurons)):
            if i == 0:
                mlp.append(nn.Linear(mlp_input_dim, mlp_neurons[i]))
            else:
                mlp.append(nn.Linear(mlp_neurons[i-1], mlp_neurons[i]))
            mlp.append(GevActivation())
            mlp.append(nn.Dropout(dropout_rate))
            if mlp_neurons[i] > 1:
                mlp.append(nn.BatchNorm1d(mlp_neurons[i]))
        mlp.append(nn.Linear(mlp_neurons[-1], 1))
        mlp.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        if x.size(0) == 1:
            # If batch size is 1, repeat the input to create a fake batch
            x = x.repeat(2, 1)
        
        importance = self.sofnn(x)
        selected_input = importance * x
        encoded, decoded = self.autoencoder(x)

        euclid_dist = torch.mean((x - decoded) ** 2, dim=1, keepdim=True)
        cos_dist = torch.sum(x * decoded, dim=1, keepdim=True) / (
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) * torch.sqrt(torch.sum(decoded ** 2, dim=1, keepdim=True)) + 1e-8
        )

        final_input = torch.cat((euclid_dist, cos_dist, selected_input, encoded), dim=1)
        prediction = self.mlp(final_input)

        if x.size(0) == 2 and x[0].equal(x[1]):
            # If we created a fake batch, return only the first prediction
            prediction = prediction[0].unsqueeze(0)

        return decoded, prediction

# Initialize the model
input_dim = trainX.shape[1]
encoder_layers = [64, 32]
decoder_layers = [32, 64]
sofnn_layers = [64, 32]
mlp_neurons = [64, 32]
dropout_rate = 0.5
model = GEV_NN(input_dim, encoder_layers, decoder_layers, sofnn_layers, mlp_neurons, dropout_rate)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Calculate class weights
unique_train, counts_train = np.unique(trainY.numpy(), return_counts=True)
class_weights = torch.tensor([1.0 / counts_train[0], 1.0 / counts_train[1]], device=device)
class_weights = class_weights / class_weights.sum()

# Loss functions and optimizer
mse_loss = nn.MSELoss()

def weighted_bce_loss(predictions, targets, weights):
    loss = nn.BCELoss(reduction='none')(predictions, targets)
    loss = loss * targets * weights[1] + loss * (1 - targets) * weights[0]
    return loss.mean()

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Function to get current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Training function
def train(model, train_loader, optimizer, mse_loss, device, class_weights):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        decoded, prediction = model(data)
        loss_ae = mse_loss(decoded, data)
        loss_pred = weighted_bce_loss(prediction, target, class_weights)
        loss = 0.25 * loss_ae + loss_pred
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Validation function
def validate(model, val_loader, mse_loss, device, class_weights):
    model.eval()
    total_loss = 0
    preds = []
    targets = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            decoded, prediction = model(data)
            loss_ae = mse_loss(decoded, data)
            loss_pred = weighted_bce_loss(prediction, target, class_weights)
            loss = 0.25 * loss_ae + loss_pred
            total_loss += loss.item()
            preds.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return total_loss / len(val_loader), preds, targets

# Prediction function
def predict(model, data_loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            _, prediction = model(data)
            preds.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return preds, targets

# Training loop
num_epochs = 150
best_val_loss = float('inf')
early_stopping_patience = 30
patience_counter = 0

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, mse_loss, device, class_weights)
    val_loss, val_preds, val_targets = validate(model, val_loader, mse_loss, device, class_weights)
    
    current_lr = get_lr(optimizer)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6f}')
    
    # Step the scheduler
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print('Early stopping triggered.')
            break

# Load the best model for evaluation
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate on validation set
val_preds, val_targets = predict(model, val_loader, device)
val_preds_binary = (val_preds >= 0.5).astype(int)

# Calculate evaluation metrics
brier_score = brier_score_loss(val_targets, val_preds)
auc_score = roc_auc_score(val_targets, val_preds)
f_score = f1_score(val_targets, val_preds_binary, average='weighted')
acc = accuracy_score(val_targets, val_preds_binary)
gmean = geometric_mean_score(val_targets, val_preds_binary, average='weighted')

print(f'Brier Score: {brier_score:.4f}')
print(f'AUC Score: {auc_score:.4f}')
print(f'F1 Score: {f_score:.4f}')
print(f'Accuracy: {acc:.4f}')
print(f'G-Mean: {gmean:.4f}')

# Confusion Matrix
cm = confusion_matrix(val_targets, val_preds_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

print("Confusion Matrix:")
print(cm)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(val_targets, val_preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()