# models/lstm_predictor.py

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

from lstm_model import LSTMModel  # relative import

# Configuration
DATA_PKL   = os.path.normpath(os.path.join(os.path.dirname(__file__), '../data/traffic_model_ready.pkl'))
MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), 'saved_models'))
INPUT_DAYS = 7     # history window in days
SEQ_LEN    = 96    # 96 intervals per day (15-min each)

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class LSTMPredictor:
    def __init__(self, data_pkl=DATA_PKL, models_dir=MODELS_DIR):
        # Load full traffic DataFrame
        self.df = pd.read_pickle(data_pkl)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        # Ensure Site_ID is str for consistency
        self.df['Site_ID'] = self.df['Site_ID'].astype(str)
        # Prepare output directory
        _ensure_dir(models_dir)
        self.models_dir = models_dir

    def train_all(self, epochs=5, batch_size=32, lr=1e-3):
        """
        Train & save one LSTM model per (Site_ID, Location) arm.
        """
        grouped = self.df.groupby(['Site_ID','Location'])
        for (site, loc), sub in grouped:
            # Filename: e.g. "0970__WARRIGAL_RD_N_of_HIGH_STREET_RD.pth"
            fname = f"{site}__{loc.replace(' ','_')}.pth"
            out_path = os.path.join(self.models_dir, fname)
            if os.path.exists(out_path):
                continue  # skip already-trained

            # Extract the sorted volume series
            ts = sub.sort_values('Timestamp')['Volume'].values
            window = INPUT_DAYS * SEQ_LEN
            if len(ts) < window + 1:
                print(f"⚠ Skipping {site}|{loc}: only {len(ts)} points")
                continue

            # Build sliding windows
            X_list, y_list = [], []
            for i in range(window, len(ts)):
                X_list.append(ts[i-window:i])
                y_list.append(ts[i])
            X_arr = np.stack(X_list, axis=0).astype(np.float32)    # (N, window)
            y_arr = np.array(y_list, dtype=np.float32).reshape(-1,1)  # (N,1)

            # Scale features and targets
            scaler = MinMaxScaler()
            X_flat = X_arr.reshape(-1,1)                   # (N*window,1)
            X_scaled = scaler.fit_transform(X_flat).reshape(-1, window)  # (N,window)
            y_scaled = scaler.transform(y_arr)             # (N,1)

            # Create tensors: (batch, seq_len, 1)
            X_tensor = torch.from_numpy(X_scaled).unsqueeze(-1)  # (N, window, 1)
            y_tensor = torch.from_numpy(y_scaled)                # (N, 1)

            ds = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            # Initialize model
            model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()

            # Training loop
            model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                for xb, yb in loader:
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg = total_loss / len(loader)
                print(f"[{site}|{loc}] Epoch {epoch+1}/{epochs} ‒ loss: {avg:.4f}")

            # Save model state and scaler
            torch.save({
                'state_dict': model.state_dict(),
                'scaler': scaler
            }, out_path)
            print(f"✔ Saved model: {fname}")

    def predict(self, site, loc, timestamp):
        """
        Predicts traffic volume (vehicles/hour) for a given site & arm at a specific timestamp.
        """
        key = f"{site}__{loc.replace(' ','_')}.pth"
        path = os.path.join(self.models_dir, key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model for {site}|{loc}")

        ckpt = torch.load(path)
        model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        scaler = ckpt['scaler']

        # Build history window ending just before `timestamp`
        ts = pd.to_datetime(timestamp)
        sub = self.df[
            (self.df['Site_ID']==site) &
            (self.df['Location']==loc) &
            (self.df['Timestamp'] < ts)
        ].sort_values('Timestamp').tail(INPUT_DAYS*SEQ_LEN)

        if len(sub) < INPUT_DAYS*SEQ_LEN:
            raise ValueError(f"Not enough history for {site}|{loc} at {timestamp}")

        seq = sub['Volume'].values.astype(np.float32).reshape(-1,1)
        seq_scaled = scaler.transform(seq)                     # (window,1)
        x = torch.from_numpy(seq_scaled).unsqueeze(0)          # (1,window,1)

        with torch.no_grad():
            pred_scaled = model(x).item()
        pred = scaler.inverse_transform([[pred_scaled]])[0][0]
        return float(pred)
