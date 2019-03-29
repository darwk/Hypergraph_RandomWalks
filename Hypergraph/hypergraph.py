import numpy as np
import scipy.sparse as sp


def get_nodeid(index_map, index, nodes_count):

    if index not in index_map:
        index_map[index] = nodes_count
        nodes_count += 1

    return index_map[index], nodes_count


def get_incidence_matrix(nodes, hyperedges):

    incidence_matrix = sp.lil_matrix((len(nodes), len(hyperedges)), dtype=float)

    edges_seen = 0
    nodes_seen = 0
    index_map = {}

    for hyperedge in hyperedges:
        for node in hyperedge:
            node_id, nodes_seen = get_nodeid(index_map, node, nodes_seen)
            incidence_matrix[node_id, edges_seen] += 1
        edges_seen += 1

    return incidence_matrix.tocsr(), index_map


def get_adj_matrices(nodes, hyperedges):

    print("Getting Incidence Matrix")
    incidence_matrix, index_map = get_incidence_matrix(nodes, hyperedges)
    incidence_matrix_density = incidence_matrix.getnnz()/(incidence_matrix.shape[0]*incidence_matrix.shape[1])
    print("incidence_matrix_density - " + str(incidence_matrix_density))

    print("Calculating Inverse edge degree Matrix")
    edge_degrees = np.subtract(sp.csr_matrix.sum(incidence_matrix, axis=0), 1)
    inv_edge_degrees = sp.spdiags(np.reciprocal(edge_degrees), [0], edge_degrees.size, edge_degrees.size, format="csr")

    incidence_matrix_transpose = incidence_matrix.transpose()

    # weights_matrix = sp.identity(incidence_matrix.shape[1], dtype=int)
    # temp = incidence_matrix.dot(weights_matrix)
    # hypergraph_adj_matrix = temp.dot(inv_edge_degrees.dot(incidence_matrix_transpose))
    # graph_adj_matrix = temp.dot(incidence_matrix_transpose)

    print("Calculating Adjacency Matrices")
    hypergraph_adj_matrix = incidence_matrix.dot(inv_edge_degrees.dot(incidence_matrix_transpose))
    hypergraph_adj_matrix = hypergraph_adj_matrix - sp.spdiags(hypergraph_adj_matrix.diagonal(), [0], hypergraph_adj_matrix.shape[0],
                                                               hypergraph_adj_matrix.shape[1], format="csr")

    graph_adj_matrix = incidence_matrix.dot(incidence_matrix_transpose)
    graph_adj_matrix = graph_adj_matrix - sp.spdiags(graph_adj_matrix.diagonal(), [0], graph_adj_matrix.shape[0],
                                                     graph_adj_matrix.shape[1], format="csr")

    adj_density = hypergraph_adj_matrix.getnnz()/(hypergraph_adj_matrix.shape[0]*hypergraph_adj_matrix.shape[1])
    print("Adjacency matrices density - " + str(adj_density))
    return hypergraph_adj_matrix, graph_adj_matrix, incidence_matrix, index_map
