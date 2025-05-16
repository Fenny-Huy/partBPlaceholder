# utils/flow_to_speed.py
import math

def flow_to_speed(flow):
    # Quadratic coefficients
    a = -1.4648375
    b = 93.75
    c = -flow

    # Discriminant
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return 0  # No real solution, assume zero speed (unlikely but safe fallback)

    sqrt_d = math.sqrt(discriminant)
    v1 = (-b + sqrt_d) / (2 * a)
    v2 = (-b - sqrt_d) / (2 * a)

    # Choose the smaller positive root (congested case)
    candidates = [v for v in [v1, v2] if v > 0]
    if not candidates:
        return 0

    speed = max(candidates)

    # Cap speed at 60 km/h for low traffic
    if flow <= 351:
        speed = min(speed, 60)

    return speed
