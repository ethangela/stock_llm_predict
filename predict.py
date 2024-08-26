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




'''lstm train/test/predict from expanded top100 features'''
# data input
Xs = pd.read_csv("top_100_features_forward3.csv") #0-3723

df = pd.read_csv("SPX2.csv")
df['predict'] = df['Close'].shift(-1)
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
ys = df['Target'].iloc[1:-1].reset_index(drop=True) #0-3719

#normlise
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(Xs)

X = torch.tensor(X_normalized, dtype=torch.float32)
y = torch.tensor(ys.values, dtype=torch.float32)

# Reshape X to 3D for LSTM: (samples, timesteps, features)
X = X.view(X.shape[0], 1, X.shape[1])

# Train-test split (keeping time series order)
train_size = int(0.8 * (len(X)-4))
X_train, X_test, X_pred = X[:train_size], X[train_size:-4], X[-4:]
y_train, y_test = y[:train_size], y[train_size:]


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

# Model hyperparameters
input_size = X_train.shape[2]
hidden_size = 50
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Training the Model
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
batch_size = 32

# Convert datasets to PyTorch DataLoader for batching
train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Evaluate the Model
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze()
    y_pred = torch.round(y_pred).int().numpy()
    y_test = y_test.int().numpy()
directional_accuracy = accuracy_score(y_test, y_pred)
print(f"Directional Accuracy: {directional_accuracy:.4f}")


# predict the model
with torch.no_grad():
    y_pred = model(X_pred).squeeze()
    y_pred = torch.round(y_pred).int().numpy()
print(f"next few days: {y_pred}")




