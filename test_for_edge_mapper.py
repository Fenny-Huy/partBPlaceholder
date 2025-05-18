from algorithms.graph_builder import build_graph
from utils.edge_mapper import EdgeMapper

centroids, edges = build_graph("data/scats_complete_average.csv")
mapper = EdgeMapper("data/traffic_model_ready.pkl", "data/scats_complete.csv")

for A,B in [("4032","4321"), ("4032","4030"), ("4032","4057"), ("4032","4034")]:
    print(f"{A}→{B} →", mapper.best_arm(A, B, centroids))
