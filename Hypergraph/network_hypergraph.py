import numpy as np
import utils


def get_node_id(index_map, index, nodes_count):

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
            node_id, nodes_seen = get_node_id(index_map, node, nodes_seen)
            incidence_matrix[node_id][edges_seen] += 1
            if incidence_matrix[node_id][edges_seen] > 1:
                print(node_id)

        edges_seen += 1

    return incidence_matrix, index_map


def get_adjacency_matrix(incidence_matrix, weights_matrix):

    edge_weights = np.diag(np.subtract(np.sum(incidence_matrix, axis=0), 1))
    utils.write_matrix_to_disk("OutputFiles/edge_weights.csv", edge_weights, "%i")

    inv_edge_weights = np.linalg.inv(edge_weights)
    utils.write_matrix_to_disk("OutputFiles/inv_edge_weights.csv", inv_edge_weights, "%.4e")

    print("Calculating adjacency_matrix")
    adjacency_matrix = np.matmul(np.matmul(incidence_matrix, weights_matrix),  np.matmul(inv_edge_weights, np.transpose(incidence_matrix)))
    print("Calculated adjacency_matrix - ", adjacency_matrix.shape)

    return adjacency_matrix


def get_clique_adjacency_matrix(incidence_matrix, weights_matrix):
    print("Calculating adjacency_matrix")
    adjacency_matrix = np.matmul(np.matmul(incidence_matrix, weights_matrix), np.transpose(incidence_matrix))
    print("Calculated adjacency_matrix - ", adjacency_matrix.shape)

    return adjacency_matrix
