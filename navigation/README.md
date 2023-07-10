## Navigation Task Pipeline

1. Execute DeepExplorer to collect temporally unidirectional topological map for each room environment, see [here](../exploration).
2. Extract SIFT feature for each image (see extract_SIFT.py).
3. Call VPR to connect temporally disconnected but spatially close nodes (see VPR.py).
4. Call ActionAssigner to assign newly added edges in step 3 with corresponding actions (see reconnect_graph.py).
5. Localize the images on both start position and goal position with VPR (see VPR.py).
5. Call Dijkstra algorithm to find the shortest path on the topological graph (see dijkstra.py).

