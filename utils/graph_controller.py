import numpy as np
from sklearn.neighbors import NearestNeighbors


class GraphController:
    def __init__(self, graph_size, start, k_size, obstacles=None, elevated_regions=None, flight_altitude=0.1):
        self.graph_size = graph_size
        self.start = np.array(start)
        self.k_size = k_size
        self.obstacles = obstacles
        self.elevated_regions = elevated_regions if elevated_regions is not None else []
        self.flight_altitude = flight_altitude
        self.node_coords = None
        self.graph = Graph()
        self.dijkstra_dist = []
        self.dijkstra_prev = []

    def _point_in_convex_polygon(self, point, vertices):
        """
        Check if a point is inside a convex polygon.
        Uses cross product method - works for both clockwise and counter-clockwise vertex order.
        For a convex polygon, a point is inside if it's on the same side of all edges.
        """
        if len(vertices) < 3:
            return False
        vertices = np.array(vertices)
        point = np.array(point)
        n = len(vertices)
        
        # Collect cross products for all edges
        cross_products = []
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            # Vector from v1 to v2
            edge_vec = v2 - v1
            # Vector from v1 to point
            point_vec = point - v1
            # Cross product (z-component): positive if point is to the left of edge
            cross = edge_vec[0] * point_vec[1] - edge_vec[1] * point_vec[0]
            cross_products.append(cross)
        
        # For a convex polygon, point is inside if all cross products have the same sign
        # (all positive or all negative, depending on vertex winding order)
        if len(cross_products) == 0:
            return False
        
        # Check if all cross products have the same sign
        all_positive = all(cp > 0 for cp in cross_products)
        all_negative = all(cp < 0 for cp in cross_products)
        
        # Also allow points exactly on edges (cross == 0)
        if all_positive or all_negative:
            return True
        
        # Check if all are non-negative or all are non-positive (including zero)
        all_non_negative = all(cp >= 0 for cp in cross_products)
        all_non_positive = all(cp <= 0 for cp in cross_products)
        
        return all_non_negative or all_non_positive
    
    def _line_segment_intersects_polygon(self, p1, p2, vertices):
        """
        Check if a line segment from p1 to p2 intersects a convex polygon.
        Returns True if the segment intersects or is inside the polygon.
        """
        vertices = np.array(vertices)
        p1 = np.array(p1)
        p2 = np.array(p2)
        
        # Check if either endpoint is inside the polygon
        if self._point_in_convex_polygon(p1, vertices) or self._point_in_convex_polygon(p2, vertices):
            return True
        
        # Check if segment intersects any edge of the polygon
        n = len(vertices)
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            if self._line_segments_intersect(p1, p2, v1, v2):
                return True
        return False
    
    def _line_segments_intersect(self, p1, p2, p3, p4):
        """
        Check if two line segments (p1-p2 and p3-p4) intersect.
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def _is_node_valid(self, node):
        """Check if a node is valid (not in elevated regions above flight altitude)"""
        for region in self.elevated_regions:
            if region['height'] > self.flight_altitude:
                if self._point_in_convex_polygon(node, region['vertices']):
                    return False
        return True
    
    def _is_edge_valid(self, p1, p2):
        """Check if an edge is valid (doesn't intersect elevated regions above flight altitude)"""
        for region in self.elevated_regions:
            if region['height'] > self.flight_altitude:
                if self._line_segment_intersects_polygon(p1, p2, region['vertices']):
                    return False
        return True

    def generate_graph(self):
        # Generate nodes, rejecting those in elevated regions
        max_attempts = self.graph_size * 10  # Limit attempts to avoid infinite loop
        attempts = 0
        valid_nodes = []
        
        # Always include start position if valid
        start_pos = self.start.reshape(1, 2)[0]
        if self._is_node_valid(start_pos):
            valid_nodes.append(start_pos)
        else:
            # If start is invalid, generate a new one
            while attempts < max_attempts:
                start_pos = np.random.rand(2)
                if self._is_node_valid(start_pos):
                    valid_nodes.append(start_pos)
                    break
                attempts += 1
        
        # Generate remaining nodes
        while len(valid_nodes) < self.graph_size + 1 and attempts < max_attempts:
            candidate = np.random.rand(2)
            if self._is_node_valid(candidate):
                valid_nodes.append(candidate)
            attempts += 1
        
        if len(valid_nodes) < self.graph_size + 1:
            print(f"Warning: Only generated {len(valid_nodes)} valid nodes out of {self.graph_size + 1} requested")
        
        self.node_coords = np.array(valid_nodes)

        if self.obstacles is None:
            X = self.node_coords
            if len(X) < 2:
                print("Warning: Not enough nodes to create graph")
                return X, self.graph.edges
            
            # Calculate max possible neighbors (excluding self)
            max_possible_neighbors = len(X) - 1
            if max_possible_neighbors < self.k_size:
                print(f"Warning: Only {len(X)} nodes available, cannot guarantee {self.k_size} neighbors per node")
            
            # Query more neighbors than k_size to account for invalid edges
            # Query up to 3x k_size or all available neighbors, whichever is smaller
            n_neighbors_to_query = min(3 * self.k_size, len(X) - 1)
            knn = NearestNeighbors(n_neighbors=n_neighbors_to_query + 1)  # +1 because it includes self
            knn.fit(X)
            distances, indices = knn.kneighbors(X)
            
            # For each node, ensure it has at least k_size valid neighbors
            for i, p in enumerate(X):
                node_str = str(i)
                self.graph.add_node(node_str)
                valid_neighbors_count = 0
                
                # First, try all queried KNN neighbors in order of distance (skip first which is self)
                for j, neighbour_idx in enumerate(indices[i][1:]):  # Skip first (self)
                    if valid_neighbors_count >= self.k_size:
                        break  # Already have enough valid neighbors
                    
                    neighbour = X[neighbour_idx]
                    # Check if edge is valid (doesn't intersect elevated regions)
                    if self._is_edge_valid(p, neighbour):
                        neighbour_str = str(neighbour_idx)
                        self.graph.add_edge(node_str, neighbour_str, distances[i, j + 1])  # j+1 because we skipped first
                        valid_neighbors_count += 1
                
                # If we still don't have enough neighbors after checking all queried neighbors,
                # try all remaining nodes in the graph (sorted by distance)
                if valid_neighbors_count < self.k_size:
                    # Get all node indices except current node and already-tried neighbors
                    tried_indices = set(indices[i][1:])  # All neighbors we already queried from KNN
                    all_other_nodes = [idx for idx in range(len(X)) if idx != i and idx not in tried_indices]
                    
                    # Try remaining nodes in order of distance
                    remaining_distances = [np.linalg.norm(X[idx] - p) for idx in all_other_nodes]
                    sorted_remaining = sorted(zip(all_other_nodes, remaining_distances), key=lambda x: x[1])
                    
                    for neighbour_idx, dist in sorted_remaining:
                        if valid_neighbors_count >= self.k_size:
                            break
                        
                        neighbour = X[neighbour_idx]
                        if self._is_edge_valid(p, neighbour):
                            neighbour_str = str(neighbour_idx)
                            self.graph.add_edge(node_str, neighbour_str, dist)
                            valid_neighbors_count += 1
                
                # Final check: warn if still insufficient (may happen with many elevated regions)
                if valid_neighbors_count < self.k_size:
                    print(f"Warning: Node {i} only has {valid_neighbors_count} valid neighbors (requested {self.k_size})")
            
            self.calc_all_path_cost()
        else:
            raise NotImplementedError
        # import matplotlib.pyplot as plt
        # plt.figure(dpi=200)
        # graph = [list(map(int, node)) for node in self.graph.edges.values()]
        # for i in range(101):
        #     for j in range(9):
        #         plt.plot([X[i, 0], X[graph[i][j + 1], 0]], [X[i, 1], X[graph[i][j + 1], 1]], c='gray', alpha=0.2)
        # for i in range(101):
        #     plt.scatter(X[i, 0], X[i, 1], c='gray', s=30)
        # plt.xlim(-0.1, 1.1)
        # plt.ylim(-0.1, 1.1)
        # plt.axis('equal')
        # plt.show()
        return X, self.graph.edges

    def find_node_index(self, p):
        if self.obstacles is None:
            X = self.node_coords
        else:
            raise NotImplementedError
        return np.where(np.linalg.norm(X - p, axis=1) < 1e-5)[0][0]

    def find_point_from_node(self, n):
        if self.obstacles is None:
            X = self.node_coords
        else:
            raise NotImplementedError
        return X[int(n)]

    def calc_all_path_cost(self):
        for coord in self.node_coords:
            start_node = str(self.find_node_index(coord))
            dist, prev = dijkstra(self.graph, start_node)
            self.dijkstra_dist.append(dist)
            self.dijkstra_prev.append(prev)

    def calc_distance(self, current, destination):
        start_node = str(self.find_node_index(current))
        end_node = str(self.find_node_index(destination))
        if start_node == end_node:
            return 0
        # path_to_end = to_array(self.dijkstra_prev[int(start_node)], end_node)
        # if len(path_to_end) <= 1: # not expand this node
        #     return 1000
        distance = self.dijkstra_dist[int(start_node)][end_node]
        distance = 0 if distance is None else distance
        return distance


class Edge:
    def __init__(self, to_node, length):
        self.to_node = to_node
        self.length = length


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, from_node, to_node, length):
        edge = Edge(to_node, length)
        if from_node in self.edges:
            from_node_edges = self.edges[from_node]
        else:
            self.edges[from_node] = dict()
            from_node_edges = self.edges[from_node]
        from_node_edges[to_node] = edge


def dijkstra(graph, source):
    q = set()
    dist = {}
    prev = {}
    for v in graph.nodes:       # initialization
        dist[v] = float('Infinity')      # unknown distance from source to v
        prev[v] = float('Infinity')      # previous node in optimal path from source
        q.add(v)                # all nodes initially in q (unvisited nodes)

    # distance from source to source
    dist[source] = 0
    while q:
        # node with the least distance selected first
        u = min_dist(q, dist)
        q.remove(u)
        try:
            if u in graph.edges:
                for _, v in graph.edges[u].items():
                    alt = dist[u] + v.length
                    if alt < dist[v.to_node]:
                        # a shorter path to v has been found
                        dist[v.to_node] = alt
                        prev[v.to_node] = u
        except:
            pass
    return dist, prev


def min_dist(q, dist):
    """
    Returns the node with the smallest distance in q.
    Implemented to keep the main algorithm clean.
    """
    min_node = None
    for node in q:
        if min_node is None:
            min_node = node
        elif dist[node] < dist[min_node]:
            min_node = node
    return min_node


def to_array(prev, from_node):
    """Creates an ordered list of labels as a route."""
    previous_node = prev[from_node]
    route = [from_node]

    while previous_node != float('Infinity'):
        route.append(previous_node)
        temp = previous_node
        previous_node = prev[temp]

    route.reverse()
    return route


if __name__ == '__main__':
    graph_ctrl = GraphController(100, [0.5,0.5], 10)
    _, graph = graph_ctrl.generate_graph()
    edge_inputs = []
    for node in graph.values():
        node_edges = list(map(int, node))
        edge_inputs.append(node_edges)
    print(edge_inputs)