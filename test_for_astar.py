from algorithms.graph_builder import build_graph
from algorithms.astar_search import astar

centroids, edges = build_graph("data/scats_complete_average.csv")

def get_dummy_volume(start, end):
    return 500  # constant for now

path, total_time = astar("3804", "4324", centroids, edges, get_dummy_volume)

print(f"\n🛣️  Path: {' → '.join(path)}")
print(f"⏱️  Total Estimated Travel Time: {total_time:.2f} minutes")