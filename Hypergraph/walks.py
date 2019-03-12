import numpy as np
import random


def get_random_walk(adj_matrix, walk_length, start, index_map, inv_index_map):

    path = []
    path.append(start)

    while len(path) < walk_length:

        adjmatrix_index = index_map[path[-1]]
        adj = adj_matrix[adjmatrix_index, :].tolil()

        next_node = np.random.choice(adj.rows[0], replace=True, p=adj.data[0]/np.sum(adj.data[0]))
        path.append(inv_index_map[next_node])

    return path


def build_deepwalk_corpus(nodes, adj_matrix, index_map, num_walks, walk_length):

    walks = []
    rand = random.Random(0)

    inv_index_map = {}
    for index in index_map:
        inv_index_map[index_map[index]] = index

    for count in range(num_walks):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(get_random_walk(adj_matrix, walk_length, node, index_map, inv_index_map))

    return walks
