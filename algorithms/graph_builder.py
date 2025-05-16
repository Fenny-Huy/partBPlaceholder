# algorithms/graph_builder.py

import pandas as pd
from utils.geo_utils import haversine

def build_graph(node_csv_path):
    """
    Build a directed graph from the centroid node file.
    Returns:
      - centroids: dict[site_id] = (lat, lon)
      - edges: list of (A, B, distance_km)
    """
    df = pd.read_csv(node_csv_path)
    print(f"🔍 Loaded {len(df)} nodes from {node_csv_path}")

    # Centroid lookup
    centroids = {
        str(row.Site_ID): (row.Latitude, row.Longitude)
        for row in df.itertuples()
    }

    # Parse roads (Location field) into lists
    df['Roads'] = df['Location'].str.split('/').apply(lambda roads: [r.strip() for r in roads])

    edges = []
    # For each road, link consecutive sites
    for road, grp in df.explode('Roads').groupby('Roads'):
        pts = grp[['Site_ID', 'Latitude', 'Longitude']].drop_duplicates()
        # Sort by predominant axis
        if pts['Latitude'].std() > pts['Longitude'].std():
            pts = pts.sort_values('Latitude', ascending=False)
        else:
            pts = pts.sort_values('Longitude')

        ids = [str(i) for i in pts['Site_ID'].tolist()]
        for i in range(len(ids) - 1):
            A, B = ids[i], ids[i+1]
            latA, lonA = centroids[A]
            latB, lonB = centroids[B]
            d = haversine(latA, lonA, latB, lonB)
            # Add both directions
            edges.append((A, B, d))
            edges.append((B, A, d))

    print(f"🔗 Built {len(edges)} directed edges across all roads.")
    return centroids, edges
