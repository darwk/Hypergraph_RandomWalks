import numpy as np
import random


def get_random_walk(adjacency_matrix, path_length, start, index_map, nodes):

    path = []
    path.append(str(start))

    while len(path) < path_length:

        adjmatrix_index = index_map[int(path[-1])]
        adj = adjacency_matrix[adjmatrix_index]

        next_node = np.random.choice(len(adj), replace=True, p=adj/np.sum(adj))
        path.append(str(nodes[next_node]))

    return path


def build_deepwalk_corpus(nodes, adjacency_matrix, num_paths, path_length, index_map):

    walks = []
    rand = random.Random(0)

    for count in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(get_random_walk(adjacency_matrix, path_length, node, index_map, nodes))

    return walks
