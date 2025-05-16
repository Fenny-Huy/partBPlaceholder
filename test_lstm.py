from utils.edge_mapper import EdgeMapper
mapper = EdgeMapper("data/traffic_model_ready.pkl")


# example in a REPL or small script
from algorithms.graph_builder import build_graph
from utils.edge_mapper import EdgeMapper

centroids, edges = build_graph("data/scats_complete_average.csv")
mapper = EdgeMapper("data/traffic_model_ready.pkl")

tests = [('4063','3122'), ('4057','4030')]
for A,B in tests:
    loc = mapper.best_arm(A, B, centroids)
    print(f"Edge {A}→{B} uses arm: {loc}")
