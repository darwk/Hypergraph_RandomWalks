import numpy as np


def get_nodeid(index_map, index, nodes_count):

    if index not in index_map:
        index_map[index] = nodes_count
        nodes_count += 1

    return index_map[index], nodes_count


def get_incidence_matrix(nodes, hyperedges):

    incidence_matrix = np.zeros((len(nodes), len(hyperedges)), dtype=int)

    edges_seen = 0
    nodes_seen = 0
    index_map = {}

    for hyperedge in hyperedges:
        for node in hyperedge:
            node_id, nodes_seen = get_nodeid(index_map, node, nodes_seen)
            incidence_matrix[node_id][edges_seen] += 1
            if incidence_matrix[node_id][edges_seen] > 1:
                print(node_id)

        edges_seen += 1

    return incidence_matrix, index_map


def get_adj_matrices(nodes, hyperedges):

    incidence_matrix, index_map = get_incidence_matrix(nodes, hyperedges)
    weights_matrix = np.identity(incidence_matrix.shape[1], dtype=int)

    edge_degrees = np.diag(np.subtract(np.sum(incidence_matrix, axis=0), 1))
    inv_edge_degrees = np.linalg.inv(edge_degrees)

    temp = np.matmul(incidence_matrix, weights_matrix)

    hypergraph_adj_matrix = np.matmul(temp,  np.matmul(inv_edge_degrees, np.transpose(incidence_matrix)))
    graph_adj_matrix = np.matmul(temp, np.transpose(incidence_matrix))

    return hypergraph_adj_matrix, graph_adj_matrix, index_map
