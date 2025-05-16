# utils/data_loader.py

import pandas as pd

def load_volume_data(pkl_path):
    """
    Load the traffic-volume time series from a pickle.
    Returns a DataFrame with columns:
      - Site_ID
      - Location
      - Latitude, Longitude
      - Timestamp (as datetime)
      - Volume (int)
    """
    df = pd.read_pickle(pkl_path)
    # Ensure Timestamp is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

def load_node_data(node_csv_path):
    """
    Load the SCATS node centroids.
    Returns a DataFrame with columns:
      - Site_ID
      - Location (road names joined by '/')
      - Latitude, Longitude (centroid of arms)
    """
    df = pd.read_csv(node_csv_path)
    return df
