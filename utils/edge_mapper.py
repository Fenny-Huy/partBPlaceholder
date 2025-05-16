# utils/edge_mapper.py

import pandas as pd
import numpy as np
from math import radians, sin, cos, atan2, degrees

def bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing from point1 → point2 in degrees [0,360).
    """
    φ1, φ2 = radians(lat1), radians(lat2)
    Δλ = radians(lon2 - lon1)
    x = sin(Δλ) * cos(φ2)
    y = cos(φ1)*sin(φ2) - sin(φ1)*cos(φ2)*cos(Δλ)
    θ = atan2(x, y)
    return (degrees(θ) + 360) % 360

class EdgeMapper:
    def __init__(self, volume_pkl):
        # Load raw volume DataFrame
        df = pd.read_pickle(volume_pkl)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Site_ID']   = df['Site_ID'].astype(str)

        # Keep one record per arm, but ensure lat/lon are floats
        arms = df[['Site_ID','Location','Latitude','Longitude']].drop_duplicates()
        arms['Latitude']  = arms['Latitude'].astype(float)
        arms['Longitude'] = arms['Longitude'].astype(float)

        self.arms = arms.reset_index(drop=True)

    def best_arm(self, A, B, centroids):
        """
        Given directed edge A→B, find the arm (Location) at A whose coordinate
        best aligns with the bearing toward B.
        """
        # Ensure A and B are strings
        A, B = str(A), str(B)

        # Centroid coords
        a_lat, a_lon = centroids[A]
        b_lat, b_lon = centroids[B]
        desired = bearing(a_lat, a_lon, b_lat, b_lon)

        # Candidate arms at site A
        candidates = self.arms[self.arms['Site_ID'] == A]
        if candidates.empty:
            raise ValueError(f"No arms found for site {A}")

        best_loc, best_diff = None, 360.0
        for _, row in candidates.iterrows():
            arm_lat, arm_lon = row['Latitude'], row['Longitude']
            arm_b = bearing(a_lat, a_lon, arm_lat, arm_lon)
            diff = abs((arm_b - desired + 180) % 360 - 180)
            if diff < best_diff:
                best_diff = diff
                best_loc  = row['Location']

        return best_loc
