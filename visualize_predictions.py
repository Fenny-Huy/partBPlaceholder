import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from models.predictor import LSTMPredictor, GRUPredictor
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel

# Parameters
SITE = '970'
LOCATION = 'WARRIGAL_RD N of HIGH STREET_RD'
SEQ_LEN = 96 * 1  # 1 day of 15-min intervals

# Load dataset
df = pd.read_pickle('data/traffic_model_ready.pkl')
df = df[(df['Site_ID'] == SITE) & (df['Location'] == LOCATION)].sort_values('Timestamp')
print(f"Loaded {len(df)} records for {SITE} | {LOCATION}")

# Extract time series
ts = df['Volume'].values
timestamps = df['Timestamp'].values

# Build sliding windows
X_list, y_list = [], []
for i in range(SEQ_LEN, len(ts)):
    X_list.append(ts[i-SEQ_LEN:i])
    y_list.append(ts[i])

X_arr = np.stack(X_list, axis=0).astype(np.float32)
y_arr = np.array(y_list).astype(np.float32).reshape(-1, 1)
timestamps = timestamps[SEQ_LEN:]

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_arr.reshape(-1, 1)).reshape(-1, SEQ_LEN)
y_scaled = scaler.transform(y_arr)

X_tensor = torch.from_numpy(X_scaled).unsqueeze(-1).float()  # (N, SEQ_LEN, 1)

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LSTM
lstm_model_path = f'lstm_saved_models/{SITE}__{LOCATION.replace(" ", "_")}.pth'
lstm_checkpoint = torch.load(lstm_model_path, map_location=device, weights_only = False)
lstm_model = LSTMModel(input_size=1, hidden_size=64, num_layers=2).to(device)
lstm_model.load_state_dict(lstm_checkpoint['state_dict'])
lstm_model.eval()

# GRU
gru_model_path = f'gru_saved_models/{SITE}__{LOCATION.replace(" ", "_")}_GRU.pth'
gru_checkpoint = torch.load(gru_model_path, map_location=device, weights_only = False)
gru_model = GRUModel(input_size=1, hidden_size=64, num_layers=2).to(device)
gru_model.load_state_dict(gru_checkpoint['state_dict'])
gru_model.eval()

# Predict
with torch.no_grad():
    lstm_preds = lstm_model(X_tensor.to(device)).cpu().numpy()
    gru_preds = gru_model(X_tensor.to(device)).cpu().numpy()

# Unscale predictions
lstm_preds_inv = scaler.inverse_transform(lstm_preds)
gru_preds_inv = scaler.inverse_transform(gru_preds)
y_actual_inv = scaler.inverse_transform(y_scaled)

# Plot
plt.figure(figsize=(15,6))
plt.plot(timestamps, y_actual_inv, label='Actual', color='black')
plt.plot(timestamps, lstm_preds_inv, label='LSTM Prediction', linestyle='--')
plt.plot(timestamps, gru_preds_inv, label='GRU Prediction', linestyle=':')
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.title(f"Traffic Volume Prediction for {SITE} | {LOCATION}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
