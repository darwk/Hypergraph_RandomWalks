import numpy as np


def write_matrix_to_disk(file_name, matrix, fmt):
    np.savetxt(file_name, matrix, fmt=fmt, delimiter=",")
    print("Successfully stored matrix in " + file_name)


def write_map_to_disk(file_name, map):
    file_handle = open(file_name, "w")

    for key in map:
        file_handle.write(str(key) + " " + str(map[key]) + "\n")

    file_handle.close()
    print("Successfully stored map in " + file_name)


def load_matrix(filepath, delimiter, dtype):
    print("Loading matrix from " + filepath)
    file = np.loadtxt(filepath, delimiter=delimiter, dtype=dtype)
    print("Successfully loaded matrix from " + filepath)
    return file


def load_map(filepath, key, value):
    print("Loading map from " + filepath)
    file_handle = open(filepath, mode='r')
    map = {}
    for line in file_handle:
        temp = line.split(' ')
        map[temp[0]] = int(temp[1])

    print("Successfully loaded map from " + filepath)
    return map


def write_adjlist(file_name, adjlist):
    file_handle = open(file_name, "w")

    for node in adjlist:
        file_handle.write(str(node) + " - ")
        for adjnode in adjlist[node]:
            file_handle.write(str(adjnode) + " ")
        file_handle.write("\n")

    file_handle.close()
    print("Successfully stored adjacency list")


def write_hyperedges(file_name, hyperedges):
    file_handle = open(file_name, "w")

    for edge in hyperedges:
        for node in edge:
            file_handle.write(str(node) + " ")
        file_handle.write("\n")

    file_handle.close()
    print("Successfully stored hyperedges")


def write_random_waks(file_name, random_walks):
    file_handle = open(file_name, "w")

    for walk in random_walks:
        for node in walk:
            file_handle.write(str(node) + " ")
        file_handle.write("\n")

    file_handle.close()
    print("Successfully stored random walks")


def load_index_map(filepath):
    print("Loading index map")

    index_map = {}
    nodes = []
    file_handle = open(filepath, mode='r')

    for line in file_handle:
        temp = line.split(' ')
        index_map[int(temp[0])] = int(temp[1])
        nodes.append(int(temp[0]))

    print("Successfully loaded index map")
    return index_map, nodes
