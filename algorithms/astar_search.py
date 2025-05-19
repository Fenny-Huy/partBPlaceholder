# algorithms/astar_search.py

import heapq
from utils.geo_utils import haversine
from utils.flow_to_speed import flow_to_speed
from utils.edge_mapper import EdgeMapper


mapper = EdgeMapper(
  arms_pkl="data/traffic_model_ready.pkl",
  nodes_csv="data/scats_complete.csv"
)


def astar(start, goal, centroids, edges, predictor, timestamp):
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
            print(f"Exploring edge {A}→{B} ({dist_km:.2f} km)")
            loc = mapper.best_arm(A, B, centroids)
            print(f"Best arm: {loc}")
            flow  = predictor.predict(A, loc, timestamp)
            speed_kmh = flow_to_speed(flow)
            travel_time = dist_km / speed_kmh * 60 + 1/2  # minutes
            print(f"speed_kmh: {speed_kmh:.2f} km/h, travel_time: {travel_time:.2f} min, volume: {flow:.2f} veh/h, loc: {loc}")

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
