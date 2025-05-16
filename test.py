
from algorithms.graph_builder import build_graph
# Adjust the path if needed
centroids, edges = build_graph("data/scats_complete_average.csv")

print("\n▶️ Sample centroids (first 5):")
for site, coord in list(centroids.items())[:5]:
    print(f"  Site {site}: {coord}")

print("\n▶️ Sample edges (first 10):")
for edge in edges:
    A, B, dist = edge
    if dist > 3:
        print(f"  {A} → {B}, {dist:.3f} km")

print(f"\n✅ Total nodes: {len(centroids)}")
print(f"✅ Total directed edges: {len(edges)}")


