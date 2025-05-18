# utils/edge_mapper.py

import pandas as pd
import re
from math import radians, sin, cos, atan2, degrees

from utils.geo_utils import bearing

# 8‐point compass directions
COMPASS_DIRS = ['N','NE','E','SE','S','SW','W','NW']

def bearing_to_compass(bearing_deg: float) -> str:
    """Map a bearing in degrees to one of N, NE, E, SE, S, SW, W, NW."""
    idx = int((bearing_deg + 22.5) // 45) % 8
    return COMPASS_DIRS[idx]

def parse_arm_location(loc_str: str):
    """
    Parse an arm Location like "WARRIGAL RD N OF RIVERSDALE RD" into:
      (main_road, arm_direction)
    """
    m = re.match(r'(.+?)\s+(N|S|E|W|NE|NW|SE|SW)\s+OF\s+(.+)', loc_str.upper())
    if not m:
        return None, None
    main_road  = m.group(1).strip()
    arm_dir    = m.group(2).strip()
    return main_road, arm_dir

class EdgeMapper:
    def __init__(self,
                 arms_pkl: str = "data/traffic_model_ready.pkl",
                 nodes_csv: str = "data/scats_complete.csv"):
        # --- Load all arms (one per row in your pickle) ---
        df = pd.read_pickle(arms_pkl)
        df['Site_ID'] = df['Site_ID'].astype(str)
        arms = (
            df[['Site_ID','Location','Latitude','Longitude']]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # ensure numeric coords
        arms['Latitude']  = arms['Latitude'].astype(float)
        arms['Longitude'] = arms['Longitude'].astype(float)
        self.arms = arms

        # --- Load per-node road lists from SCATS metadata ---
        meta = pd.read_csv(nodes_csv, dtype={'SCATS_Number': str})
        meta = meta.rename(columns={'SCATS_Number':'Site_ID'})
        meta['Site_ID'] = meta['Site_ID'].astype(str)
        # split the "Location Description" by "/" to get all roads at that intersection
        meta['Roads'] = meta['Location Description'].str.upper().str.split('/')
        self.node_roads = dict(zip(meta['Site_ID'], meta['Roads']))

    def best_arm(self, A: str, B: str, centroids: dict) -> str:
        """
        For edge A→B, pick the correct arm Location at A:
          1) main_road ∈ B’s road list
          2) arm_direction == compass_dir(A→B)
        Fallback: minimal bearing difference among candidates.
        """
        A, B = str(A), str(B)
        if A not in self.node_roads or B not in self.node_roads:
            raise KeyError(f"Missing node_roads for A={A} or B={B}")

        # 1) Compute bearing and compass direction
        a_lat, a_lon = centroids[A]
        b_lat, b_lon = centroids[B]
        true_br = bearing(a_lat, a_lon, b_lat, b_lon)
        comp_dir = bearing_to_compass(true_br)

        # 2) Filter arms at A whose main_road is in B’s roads
        b_roads = set(self.node_roads[B])
        cands = []
        for _, row in self.arms[self.arms['Site_ID']==A].iterrows():
            loc = row['Location'].upper()
            main_road, arm_dir = parse_arm_location(loc)
            if main_road and main_road in b_roads:
                cands.append((loc, arm_dir, row['Latitude'], row['Longitude']))

        # 3) If none matched the road list, consider all arms at A
        if not cands:
            for _, row in self.arms[self.arms['Site_ID']==A].iterrows():
                loc = row['Location'].upper()
                main_road, arm_dir = parse_arm_location(loc)
                cands.append((loc, arm_dir, row['Latitude'], row['Longitude']))

        # 4) Among candidates, keep only those with matching arm_dir
        matched = [c for c in cands if c[1] == comp_dir]
        if matched:
            cands = matched

        # 5) If still multiple (or none), pick by minimal bearing diff
        best_loc = None
        best_diff = 360.0
        for loc, arm_dir, lat, lon in cands:
            arm_br = bearing(a_lat, a_lon, lat, lon)
            diff = abs((arm_br - true_br + 180) % 360 - 180)
            if diff < best_diff:
                best_diff = diff
                best_loc  = loc

        return best_loc
