from algorithms.graph_builder import build_graph

centroids, edges = build_graph("data/scats_complete_average.csv")
print(f"edges: {edges}")