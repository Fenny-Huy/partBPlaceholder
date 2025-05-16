# algorithms/astar_search.py

import heapq
from utils.geo_utils import haversine
from utils.flow_to_speed import flow_to_speed
from utils.edge_mapper import EdgeMapper


mapper = EdgeMapper("data/traffic_model_ready.pkl")


def astar(start, goal, centroids, edges, get_volume_at_edge, predictor, timestamp):
    """
    A* search using predicted volume to estimate travel time.

    get_volume_at_edge(start_node, end_node) → predicted volume at that edge
    """
    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while frontier:
        current_f, current = heapq.heappop(frontier)

        if current == goal:
            break

        for (A, B, dist_km) in [e for e in edges if e[0] == current]:
            loc = mapper.best_arm(A, B, centroids)
            flow  = predictor.predict(A, loc, timestamp)
            speed_kmh = flow_to_speed(flow)
            travel_time = dist_km / speed_kmh * 60  # minutes
            print(f"speed_kmh: {speed_kmh:.2f} km/h, travel_time: {travel_time:.2f} min, dist_km: {dist_km:.2f} km, volume: {flow:.2f} veh/h, loc: {loc}, A: {A}, B: {B}")

            new_cost = cost_so_far[current] + travel_time
            if B not in cost_so_far or new_cost < cost_so_far[B]:
                cost_so_far[B] = new_cost
                priority = new_cost + haversine(*centroids[B], *centroids[goal])
                heapq.heappush(frontier, (priority, B))
                came_from[B] = current

    # Reconstruct path
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from.get(node)
        if node is None:
            return [], float('inf')  # No path
    path.append(start)
    path.reverse()

    return path, cost_so_far[goal]
