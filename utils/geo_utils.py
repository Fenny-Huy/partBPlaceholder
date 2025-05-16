# utils/geo_utils.py

from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance (km) between two points:
      (lat1, lon1) and (lat2, lon2), in decimal degrees.
    """
    R = 6371.0  # Earth radius in km
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ / 2)**2 + cos(φ1) * cos(φ2) * sin(Δλ / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
