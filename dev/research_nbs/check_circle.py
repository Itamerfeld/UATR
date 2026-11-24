# import networkx as nx
# import numpy as np
# from scipy.sparse.linalg import eigsh

# def check_circular_embedding(G, tolerance=0.15):
#     """
#     Validates fuzzy ring topology using Spectral Embedding and 
#     distance variance analysis.
#     """
#     if not nx.is_connected(G):
#         return False, 0.0

#     # 1. Compute Laplacian and Eigen-decomposition
#     # We need the first 3 eigenvectors (0, 1, 2)
#     L = nx.laplacian_matrix(G).asfptype()
#     vals, vecs = eigsh(L, k=3, which='SM')
    
#     # 2. Project into 2D using the two smallest non-zero eigenvectors
#     # vecs[:, 0] is the constant eigenvector for lambda=0, so we use 1 and 2
#     x = vecs[:, 1]
#     y = vecs[:, 2]
#     coords = np.column_stack((x, y))
    
#     # 3. Calculate distances from the centroid
#     centroid = np.mean(coords, axis=0)
#     distances = np.linalg.norm(coords - centroid, axis=1)
    
#     # 4. Quantify 'Circularness'
#     # A perfect ring has a Coefficient of Variation (CV) near 0.
#     mean_dist = np.mean(distances)
#     std_dist = np.std(distances)
#     cv = std_dist / mean_dist
    
#     # A 'fuzzy' ring typically has a CV < 0.2 depending on noise levels
#     is_ring = cv < tolerance
    
#     return is_ring, cv

# # Example Usage:
# # G = nx.watts_strogatz_graph(n=100, k=4, p=0.1) # Small-world (Fuzzy Ring)
# # is_ring, score = check_circular_embedding(G)

# def analyze_fuzzy_ring_with_outliers(G):
#     # 1. Identify connected components
#     components = sorted(nx.connected_components(G), key=len, reverse=True)
#     lcc_nodes = components[0]
    
#     # Calculate what percentage of the graph is in the main mass
#     mass_ratio = len(lcc_nodes) / len(G)
    
#     # 2. Extract the subgraph of the main mass
#     G_main = G.subgraph(lcc_nodes).copy()
    
#     # 3. Run the Embedding Check on the main mass only
#     is_ring, score = check_circular_embedding(G_main)
    
#     return {
#         "is_main_mass_ring": is_ring,
#         "ring_quality_score": score,
#         "main_mass_percentage": mass_ratio,
#         "outlier_count": len(G) - len(lcc_nodes)
#     }

# def get_lcc_eigenvalues_from_full_laplacian(G, k_search=10):
#     # 1. Identify LCC and its node indices
#     components = sorted(nx.connected_components(G), key=len, reverse=True)
#     lcc_nodes = list(components[0])
#     # Map node labels to matrix indices
#     node_to_idx = {node: i for i, node in enumerate(G.nodes())}
#     lcc_indices = [node_to_idx[node] for node in lcc_nodes]

#     # 2. Compute first few eigenvalues/vectors of the FULL Laplacian
#     L = nx.laplacian_matrix(G).asfptype()
#     vals, vecs = eigsh(L, k=k_search, which='SM')

#     lcc_eigenvalues = []
    
#     # 3. Filter by checking eigenvector localization
#     # We ignore the first N eigenvalues that are ~0 (one per component)
#     for i in range(len(vals)):
#         if vals[i] < 1e-10: 
#             continue # Skip the trivial 0 eigenvalues
            
#         # Calculate how much 'energy' of the eigenvector is in the LCC
#         eigenvector = vecs[:, i]
#         energy_in_lcc = np.sum(eigenvector[lcc_indices]**2)
#         total_energy = np.sum(eigenvector**2)
        
#         # If > 99% of the energy is in the LCC, it belongs to the LCC
#         if (energy_in_lcc / total_energy) > 0.99:
#             lcc_eigenvalues.append(vals[i])

#     return lcc_eigenvalues

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# Your data
adj = np.array([
    [ 0,  3, 19, 26, 12,  6,  2,  0,  0,  0],
    [ 1,  0,  5, 21, 19, 14,  7,  3,  1,  0],
    [ 0,  1,  0,  1, 22, 28,  8,  4,  1,  1],
    [ 0,  0,  1,  0,  0, 11, 26, 17,  3,  5],
    [ 4,  2,  0,  0,  0,  0, 11, 23, 15,  3],
    [ 6,  0,  0,  2,  0,  1,  1, 12, 25, 17],
    [13,  2,  1,  0,  0,  0,  0,  2, 16, 22],
    [29, 13,  1,  1,  0,  0,  0,  2,  2, 15],
    [13, 29, 16,  1,  0,  1,  1,  0,  0,  2],
    [ 2, 20, 23, 11,  5,  3,  1,  0,  0,  1]
])

def plot_weighted_ring(A):
    # 1. Symmetrize the matrix
    A_sym = (A + A.T) / 2
    
    # 2. Create the Graph and Laplacian
    G = nx.from_numpy_array(A_sym)
    L = nx.laplacian_matrix(G).astype(np.float32).toarray()
    
    # 3. Solve for Eigenvectors
    # For a 10x10, we can just use np.linalg.eigh (Dense solver)
    vals, vecs = np.linalg.eigh(L)
    
    # x and y are the 1st and 2nd non-zero eigenvectors
    x = vecs[:, 1]
    y = vecs[:, 2]
    
    # 4. Plot
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=100, c=range(10), cmap='hsv')
    
    # Draw connections based on original weights
    for i in range(10):
        for j in range(10):
            if A[i, j] > np.mean(A) + np.std(A): # Threshold to keep plot clean
                plt.annotate("", xy=(x[j], y[j]), xytext=(x[i], y[i]),
                             arrowprops=dict(arrowstyle="->", alpha=0.3))
        plt.text(x[i], y[i], str(i), fontsize=12, fontweight='bold')

    plt.title("Spectral Embedding of Weighted Directed Ring")
    plt.axis('equal')
    plt.show()

plot_weighted_ring(adj)