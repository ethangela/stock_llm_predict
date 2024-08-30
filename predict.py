import pandas as pd
import numpy as np
from itertools import combinations, product
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from chronos import ChronosPipeline
import pickle





'''lstm training fine tuning (next-day output)'''
# Build the model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.batch_norm(out[:, -1, :])  # Apply batch normalization on the last output
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# Data input
Xs = pd.read_csv("top_100_features_forward3.csv")  # 0-3723

df = pd.read_csv("SPX2.csv")
df['predict'] = df['Close'].shift(-1)
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
ys = df['Target'].iloc[1:-1].reset_index(drop=True)  # 0-3719

# Normalize
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(Xs)
X_normalized = X_normalized[:-4,:]
X_to_be_predict = X_normalized[-4:,:]

# Function to create sequences with a given backward window
def create_sequences(X, y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:i + look_back])
        ys.append(y[i + look_back])
    return np.array(Xs), np.array(ys)

# Define ranges for hyperparameters
look_back_values = [1,2,3,4,5,10,20]
hidden_size_values = [25,50,75,100]
num_layers_values = [1,2,3]

# Function to evaluate a specific set of hyperparameters
def evaluate_model(look_back, hidden_size, num_layers):
    X_seq, y_seq = create_sequences(X_normalized, ys, look_back)
    X = torch.tensor(X_seq, dtype=torch.float32)
    y = torch.tensor(y_seq, dtype=torch.float32)

    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))
    test_size = len(X) - train_size - val_size

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

    input_size = X_train.shape[2]
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    batch_size = 32

    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

    val_data = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_outputs = model(val_X)
                val_loss = criterion(val_outputs.squeeze(), val_y)
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    return best_val_loss, best_model_state

# Grid search for hyperparameter tuning
best_loss = float('inf')
best_params = None

for look_back, hidden_size, num_layers in itertools.product(look_back_values, hidden_size_values, num_layers_values):
    val_loss, model_state = evaluate_model(look_back, hidden_size, num_layers)
    if val_loss < best_loss:
        best_loss = val_loss
        best_params = (look_back, hidden_size, num_layers)
        best_model_state = model_state

print(f"Best Hyperparameters: Look-Back: {best_params[0]}, Hidden Size: {best_params[1]}, Num Layers: {best_params[2]}")











'''lstm training and testing (next-day output)'''
# Data input
Xs = pd.read_csv("top_100_features_forward3.csv")  # 0-3723

df = pd.read_csv("SPX2.csv")
df['predict'] = df['Close'].shift(-1)
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
ys = df['Target'].iloc[1:-1].reset_index(drop=True)  # 0-3719

# Normalize
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(Xs)

# Function to create sequences with a given backward window
def create_sequences(X, y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:i + look_back])
        ys.append(y[i + look_back])
    return np.array(Xs), np.array(ys)

# Optimized hyper-parameters
# Best Hyperparameters: Look-Back: 4, Hidden Size: 50, Num Layers: 3
look_back = 4  # Example window size
hidden_size = 50
num_layers = 3

# Build data
X_use = X_normalized[:-4,:]
X_seq, y_seq = create_sequences(X_use, ys, look_back)
X = torch.tensor(X_seq, dtype=torch.float32)
y = torch.tensor(y_seq, dtype=torch.float32)

train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.batch_norm(out[:, -1, :])  # Apply batch normalization on the last output
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# Build the model
input_size = X_train.shape[2]
output_size = 1
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Train the model
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
batch_size = 32

train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

val_data = torch.utils.data.TensorDataset(X_val, y_val)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

best_val_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs.squeeze(), val_y)
            val_losses.append(val_loss.item())
    
    avg_val_loss = np.mean(val_losses)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()

# Evaluate the model
model.load_state_dict(best_model_state)

model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze()
    y_pred = torch.round(y_pred).int().numpy()
    y_test = y_test.int().numpy()
directional_accuracy = accuracy_score(y_test, y_pred)
print(f"Directional Accuracy on test data: {directional_accuracy:.4f}")


# Using model for prediction
def create_X_sequences(X, look_back):
    Xs = []
    for i in range(len(X) - look_back):
        Xs.append(X[i:i + look_back])
    return np.array(Xs)

X_seq = create_X_sequences(X_normalized, look_back)
X = torch.tensor(X_seq, dtype=torch.float32)
X_pred = X[-4:]

with torch.no_grad():
    y_pred = model(X_pred).squeeze()
    y_pred = torch.round(y_pred).int().numpy()
print(f"next few days: {y_pred}")


