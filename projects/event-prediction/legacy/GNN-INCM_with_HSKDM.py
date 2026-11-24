# GNN-INCM with HSKDM

# Install necessary libraries
!mamba install -y -q pandas numpy scikit-learn imbalanced-learn scipy matplotlib seaborn fsspec pytorch pytorch_geometric pytorch_sparse dask torch-optimizer optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GraphConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, recall_score, confusion_matrix, 
                             precision_score, f1_score, balanced_accuracy_score, 
                             matthews_corrcoef, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from torch_geometric.utils import dropout_edge
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
training_data = pd.read_csv("training_data.csv")
validation_data = pd.read_csv("validation_data.csv")

# Preprocessing
target_column = 'event_identifier'  # Adjust as per your target variable
features = training_data.drop(['record_id', target_column], axis=1)
target = training_data[target_column]

# Combine training and validation data
all_features = pd.concat([features, validation_data.drop(['record_id', target_column], axis=1)])
all_target = pd.concat([target, validation_data[target_column]])

# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(
    all_features, all_target, test_size=0.2, random_state=42, stratify=all_target
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Balance the training data using SMOTE and Random Under-Sampling
over = SMOTE(sampling_strategy=0.5, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
pipeline = Pipeline(steps=[('o', over), ('u', under)])
X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Create PyTorch Geometric Data objects
def create_graph_data(X, y=None, k=100):
    """
    Transforms feature matrix and labels into PyTorch Geometric Data objects with k-NN graph.
    
    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (pandas.Series or numpy.ndarray, optional): Target labels.
    - k (int): Number of nearest neighbors.
    
    Returns:
    - Data: PyTorch Geometric Data object.
    """
    X_tensor = torch.FloatTensor(X)
    if y is not None:
        y_tensor = torch.LongTensor(y.values).squeeze()
    else:
        y_tensor = None
    
    # Compute pairwise distances
    dist = torch.cdist(X_tensor, X_tensor)
    
    # Get k nearest neighbors for each node
    _, indices = dist.topk(k + 1, largest=False)  # +1 to exclude self
    row = torch.arange(X_tensor.size(0)).unsqueeze(1).expand(-1, k)
    edge_index = torch.stack([row.reshape(-1), indices[:, 1:].reshape(-1)])
    
    data = Data(x=X_tensor, edge_index=edge_index)
    if y_tensor is not None:
        data.y = y_tensor
    return data

# Create graph data for train, validation, and test sets
train_data = create_graph_data(X_train_scaled, y_train_resampled)
val_data = create_graph_data(X_val_scaled, y_val)
test_data = create_graph_data(X_test_scaled, y_test)

# Define GNNLayer supporting multiple architectures
class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, layer_type='GCN'):
        """
        Initializes a GNN layer.
        
        Parameters:
        - in_channels (int): Number of input features.
        - out_channels (int): Number of output features.
        - layer_type (str): Type of GNN layer ('GCN', 'SAGE', 'GAT', 'GraphConv').
        """
        super(GNNLayer, self).__init__()
        if layer_type == 'GCN':
            self.conv = GCNConv(in_channels, out_channels)
        elif layer_type == 'SAGE':
            self.conv = SAGEConv(in_channels, out_channels)
        elif layer_type == 'GAT':
            self.conv = GATConv(in_channels, out_channels)
        elif layer_type == 'GraphConv':
            self.conv = GraphConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def forward(self, x, edge_index):
        """
        Forward pass for the GNN layer.
        
        Parameters:
        - x (Tensor): Node feature matrix.
        - edge_index (Tensor): Edge indices.
        
        Returns:
        - Tensor: Updated node features.
        """
        return self.conv(x, edge_index)

# Define the GNNINCM model
class GNNINCM(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_classes, dropout_rate=0.5, layer_types=['GCN', 'SAGE', 'GAT', 'GraphConv']):
        """
        Initializes the GNN-INCM model.
        
        Parameters:
        - input_dim (int): Number of input features.
        - hidden_dims (list of int): Hidden layer dimensions.
        - output_dim (int): Dimension of the output embedding.
        - num_classes (int): Number of target classes.
        - dropout_rate (float): Dropout rate for regularization.
        - layer_types (list of str): Types of GNN layers to use.
        """
        super(GNNINCM, self).__init__()
        self.convs = nn.ModuleList([
            GNNLayer(
                input_dim if i == 0 else hidden_dims[i-1],
                hidden_dims[i],
                layer_type=layer_types[i % len(layer_types)]
            )
            for i in range(len(hidden_dims))
        ])
        self.fc1 = nn.Linear(hidden_dims[-1], output_dim)
        self.fc2 = nn.Linear(output_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for dim in hidden_dims])

    def forward(self, x, edge_index):
        """
        Forward pass for the GNN-INCM model.
        
        Parameters:
        - x (Tensor): Node feature matrix.
        - edge_index (Tensor): Edge indices.
        
        Returns:
        - Tuple[Tensor, Tensor]: Embedding and output logits.
        """
        for conv, bn in zip(self.convs, self.bns):
            # Apply dropout to edges to prevent overfitting
            edge_index, _ = dropout_edge(edge_index, p=0.2, force_undirected=True, training=self.training)
            x = F.elu(bn(conv(x, edge_index)))
            x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        return x, output

# Define FocalLoss to handle class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Initializes the Focal Loss.
        
        Parameters:
        - alpha (Tensor, optional): Balancing factor for classes.
        - gamma (float): Focusing parameter.
        - reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss.
        
        Parameters:
        - inputs (Tensor): Model predictions.
        - targets (Tensor): Ground truth labels.
        
        Returns:
        - Tensor: Computed loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

# Define the HSKDM framework
class HSKDM:
    def __init__(self, num_models, input_dim, hidden_dims, output_dim, num_classes, device, layer_types=['GCN', 'SAGE', 'GAT', 'GraphConv']):
        """
        Initializes the HSKDM framework with an ensemble of GNNINCM models.
        
        Parameters:
        - num_models (int): Number of models in the ensemble.
        - input_dim (int): Number of input features.
        - hidden_dims (list of int): Hidden layer dimensions.
        - output_dim (int): Dimension of the output embedding.
        - num_classes (int): Number of target classes.
        - device (torch.device): Device to run the models on.
        - layer_types (list of str): Types of GNN layers to use.
        """
        self.num_models = num_models
        self.models = [
            GNNINCM(input_dim, hidden_dims, output_dim, num_classes, layer_types=layer_types).to(device)
            for _ in range(num_models)
        ]
        
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
            for model in self.models
        ]
        self.schedulers = [
            ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
            for opt in self.optimizers
        ]
        
        self.device = device
        class_counts = np.bincount(y_train_resampled)
        class_weights = torch.tensor([class_counts[1] / class_counts[0], 1.0]).to(device)  # Adjust weights based on class distribution
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=2).to(device)

    def train(self, data):
        """
        Trains the ensemble of models for one epoch.
        
        Parameters:
        - data (Data): PyTorch Geometric Data object for training.
        
        Returns:
        - float: Average loss across models.
        """
        for model in self.models:
            model.train()

        losses = []
        for i, model in enumerate(self.models):
            self.optimizers[i].zero_grad()

            _, output = model(data.x.to(self.device), data.edge_index.to(self.device))
            
            loss = self.focal_loss(output, data.y.to(self.device))
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            self.optimizers[i].step()
            
            losses.append(loss.item())

        return np.mean(losses)

    def evaluate(self, data):
        """
        Evaluates the ensemble of models and aggregates predictions.
        
        Parameters:
        - data (Data): PyTorch Geometric Data object for evaluation.
        
        Returns:
        - Tensor: Averaged prediction probabilities.
        """
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                _, output = model(data.x.to(self.device), data.edge_index.to(self.device))
                pred = F.softmax(output, dim=1)[:, 1]  # Probability for the positive class
                predictions.append(pred)
        
        # Ensemble prediction by averaging
        final_pred = torch.stack(predictions).mean(dim=0)
        return final_pred

    def update_lr(self, val_auc):
        """
        Updates the learning rate based on validation AUC.
        
        Parameters:
        - val_auc (float): Validation AUC score.
        """
        for scheduler in self.schedulers:
            scheduler.step(val_auc)

# Model initialization parameters
input_dim = X_train_scaled.shape[1]
hidden_dims = [128, 64, 32]
output_dim = 16
num_classes = 2

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize HSKDM framework with 3 models
hskdm = HSKDM(
    num_models=3, 
    input_dim=input_dim, 
    hidden_dims=hidden_dims, 
    output_dim=output_dim, 
    num_classes=num_classes, 
    device=device
)

# Training loop parameters
num_epochs = 300
best_val_auc = 0
patience = 20
epochs_no_improve = 0

train_losses = []
val_aucs = []

for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    try:
        # Train the ensemble
        loss = hskdm.train(train_data)
        train_losses.append(loss)
        
        # Evaluate on validation set
        val_pred = hskdm.evaluate(val_data)
        val_auc = roc_auc_score(val_data.y.cpu().numpy(), val_pred.cpu().numpy())
        val_aucs.append(val_auc)
        
        # Update learning rate schedulers
        hskdm.update_lr(val_auc)
        
        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            # Save the best model states
            torch.save([model.state_dict() for model in hskdm.models], 'best_models.pth')
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= patience:
            logger.info(f'Early stopping triggered at epoch {epoch}')
            break
        
        # Logging every 10 epochs
        if epoch % 10 == 0:
            logger.info(f'Epoch: {epoch}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')
    
    except RuntimeError as e:
        logger.error(f"RuntimeError at epoch {epoch}: {str(e)}")
        break
    except Exception as e:
        logger.error(f"Unexpected error at epoch {epoch}: {str(e)}")
        break

# Load best models for final evaluation
best_models_state_dicts = torch.load('best_models.pth')
for model, state_dict in zip(hskdm.models, best_models_state_dicts):
    model.load_state_dict(state_dict)

# Final evaluation on test data
def find_optimal_threshold(y_true, y_pred):
    """
    Finds the optimal threshold to convert probabilities to binary predictions based on Youden's J statistic.
    
    Parameters:
    - y_true (array-like): True binary labels.
    - y_pred (array-like): Predicted probabilities for the positive class.
    
    Returns:
    - float: Optimal threshold value.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    return thresholds[ix]

# Get predictions on test set
test_pred = hskdm.evaluate(test_data)
optimal_threshold = find_optimal_threshold(y_test, test_pred.cpu().numpy())

# Convert probabilities to binary predictions
test_pred_binary = (test_pred.cpu().numpy() > optimal_threshold).astype(int)

# Calculate evaluation metrics
test_auc = roc_auc_score(y_test, test_pred.cpu().numpy())
test_recall = recall_score(y_test, test_pred_binary)
test_precision = precision_score(y_test, test_pred_binary)
test_f1 = f1_score(y_test, test_pred_binary)
test_balanced_accuracy = balanced_accuracy_score(y_test, test_pred_binary)
test_mcc = matthews_corrcoef(y_test, test_pred_binary)

# Print evaluation results
print(f'Optimal threshold: {optimal_threshold:.4f}')
print(f'Test AUC: {test_auc:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')
print(f'Test Balanced Accuracy: {test_balanced_accuracy:.4f}')
print(f'Test Matthews Correlation Coefficient: {test_mcc:.4f}')

# Display Confusion Matrix
cm = confusion_matrix(y_test, test_pred_binary)
print("Confusion Matrix:")
print(cm)

print("\nTraining complete. Model evaluation finished.")