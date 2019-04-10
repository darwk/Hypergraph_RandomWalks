import sys
import argparse
import scipy.sparse as sp
import numpy as np

from citation_network import get_citation_network
from coreference_network import get_coreference_network
from hypergraph import get_adj_matrices
from movielens_network import get_movielens_network
from walks import build_deepwalk_corpus
from gensim.models import Word2Vec


def get_representations(nodes, adj_matrix, index_map, num_walks, walk_length, num_dimensions, window_size):
    print("Building random walk corpus")
    random_walks = build_deepwalk_corpus(nodes, adj_matrix, index_map, num_walks, walk_length)

    print("Getting representations")
    model = Word2Vec(random_walks, size=num_dimensions, window=window_size, min_count=0, sg=1, hs=1)
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["aminer", "citeseer", "cora", "movielens"],
                        help="Dataset to be used - aminer or citeseer" or "cora" or "movielens")

    parser.add_argument("--network", type=str, choices=["cocitation", "coreference"],
                        help="Network type to be fetched - cocitation or coreference")

    parser.add_argument("--num_walks", nargs='+', type=int,
                        help="number of random walks per node")

    parser.add_argument("--walk_length", nargs='+', type=int,
                        help="length of the walk")

    parser.add_argument("--num_dimensions", nargs='+', type=int,
                        help="number of dimensions")

    parser.add_argument("--window_size", nargs='+', type=int,
                        help="window size")

    parser.add_argument("--use_cc", type=str,
                        help="whether to use connected component or not")

    parser.add_argument("--p", type=float,
                        help="parameter")

    parser.add_argument("--q", type=float,
                        help="parameter")

    parser.add_argument("--output", type=str,
                        help="output folder path")

    args = parser.parse_args()
    dataset = args.dataset
    network = args.network
    num_walks_list = args.num_walks
    walk_length_list = args.walk_length
    num_dimensions_list = args.num_dimensions
    window_size_list = args.window_size
    use_cc = args.use_cc
    p = args.p
    q = args.q
    output_folder = args.output

    if network == "cocitation":
        if dataset == "movielens":
            nodes, hyperedges, paperid_classid, classid_classname = get_movielens_network(use_cc)
        else:
            nodes, hyperedges, paperid_classid, classid_classname = get_citation_network(dataset, use_cc)
    elif network == "coreference":
        nodes, hyperedges, paperid_classid, classid_classname = get_coreference_network(dataset, use_cc)

    hypergraph_adj_matrix, graph_adj_matrix, incidence_matrix, index_map = get_adj_matrices(nodes, hyperedges)

    node_degrees = np.squeeze(np.asarray(sp.csr_matrix.sum(incidence_matrix, axis=1)))

    print("Average node degree - " + str(np.sum(node_degrees)/len(node_degrees)))

    scaled_node_degrees = np.multiply(np.float_power(node_degrees, q), p)
    scaled_hypergraph_adj_matrix = hypergraph_adj_matrix.dot(sp.diags(scaled_node_degrees))

    for num_walks in num_walks_list:
        for walk_length in walk_length_list:
            for num_dimensions in num_dimensions_list:
                for window_size in window_size_list:
                    hypergraph_model = get_representations(nodes, scaled_hypergraph_adj_matrix, index_map,
                                                           num_walks, walk_length, num_dimensions, window_size)

                    hypergraph_model.wv.save_word2vec_format(output_folder + "/hypergraph_model2_" + str(p) + "_" + str(q) + "_" + str(num_walks) + "_"
                                                             + str(walk_length) + "_" + str(num_dimensions) + "_" + str(window_size) + ".txt")
                    print("Successfully got hypergraph model with purity factor")

                    graph_model = get_representations(nodes, graph_adj_matrix, index_map,
                                                      num_walks, walk_length, num_dimensions, window_size)
                    graph_model.wv.save_word2vec_format(output_folder +"/graph_model_" + str(p) + "_" + str(q) + "_" + str(num_walks) + "_"
                                                             + str(walk_length) + "_" + str(num_dimensions) + "_" + str(window_size) + ".txt")
                    print("Successfully got graph model")


if __name__ == "__main__":
  sys.exit(main())
