import numpy as np

from citation_network import get_citation_network
from network_hypergraph import get_incidence_matrix, get_adjacency_matrix, get_clique_adjacency_matrix
from utils import write_matrix_to_disk, write_random_waks, write_map_to_disk, load_matrix, load_index_map
from walks import build_deepwalk_corpus
from gensim.models import Word2Vec


def process_1():

    nodes, hyperedges, paperid_classid, classid_classname = get_citation_network("filePaths.txt")

    incidence_matrix, index_map = get_incidence_matrix(nodes, hyperedges)
    write_matrix_to_disk("OutputFiles/incidence_matrix.csv", incidence_matrix, '%i')
    write_map_to_disk("OutputFiles/index_map.txt", index_map)


def process_2(graph_type):

    incidence_matrix = load_matrix("OutputFiles/incidence_matrix.csv", delimiter=',', dtype="int")
    index_map, nodes = load_index_map("OutputFiles/index_map.txt")

    weights_matrix = np.identity(incidence_matrix.shape[1], dtype=int)

    if graph_type == "hypergraph":
        adjacency_matrix = get_adjacency_matrix(incidence_matrix, weights_matrix)
    elif graph_type == "graph":
        adjacency_matrix = get_clique_adjacency_matrix(incidence_matrix, weights_matrix)

    write_matrix_to_disk("OutputFiles/" + graph_type + "_adjacency_matrix.csv", adjacency_matrix, "%.4e")
    random_walks = build_deepwalk_corpus(nodes, adjacency_matrix, 50, 40, index_map)
    write_random_waks("OutputFiles/" + graph_type + "_random_walks.txt", random_walks)

    model = Word2Vec(random_walks, size=32, window=10, min_count=0, sg=1, hs=1, workers=1)
    model.wv.save_word2vec_format("OutputFiles/" + graph_type + "_node_embeddings.txt")


process_1()
process_2("hypergraph")
process_2("graph")
