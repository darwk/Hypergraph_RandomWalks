import sys
import argparse
import numpy as np

from citation_network import get_citation_network
from classifier import svm_classifier, knn_classifier
from hypergraph import get_adj_matrices
from walks import build_deepwalk_corpus
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


def get_representations(nodes, adj_matrix, index_map, num_walks, walk_length, num_dimensions, window_size):
    random_walks = build_deepwalk_corpus(nodes, adj_matrix, index_map, num_walks, walk_length)
    model = Word2Vec(random_walks, size=num_dimensions, window=window_size, min_count=0, sg=1, hs=1, workers=1)
    return model


def evaluate(nodes, target_classes, hypergraph_model, graph_model, num_neighbors):

    nodes_train, nodes_test, target_classes_train, target_classes_test = train_test_split(nodes, target_classes, random_state=1234)

    hypergraph_train = []
    graph_train = []

    for node in nodes_train:
        hypergraph_train.append(hypergraph_model[node])
        graph_train.append(graph_model[node])

    hypergraph_train = np.array(hypergraph_train)
    graph_train = np.array(graph_train)

    hypergraph_test = []
    graph_test = []

    for node in nodes_test:
        hypergraph_test.append(hypergraph_model[node])
        graph_test.append(graph_model[node])

    hypergraph_test = np.array(hypergraph_test)
    graph_test = np.array(graph_test)

    #svm_classifier(hypergraph_train, hypergraph_test, classes_train, classes_test, C, gamma)
    #svm_classifier(graph_train, graph_test, classes_train, classes_test, C, gamma)

    hypergraph_accuracy, hypergraph_conf_matrix = knn_classifier(hypergraph_train, hypergraph_test,
                                                                 target_classes_train, target_classes_test, num_neighbors)
    graph_accuracy, graph_conf_matrix = knn_classifier(graph_train, graph_test, target_classes_train,
                                                       target_classes_test, num_neighbors)

    return hypergraph_accuracy, graph_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_walks", nargs='+', type=int, help="number of random walks per node")
    parser.add_argument("--walk_length", nargs='+', type=int, help="length of the walk")
    parser.add_argument("--num_dimensions", nargs='+', type=int, help="number of dimensions")
    parser.add_argument("--window_size", nargs='+', type=int, help="window size")
    parser.add_argument("--num_neighbors", nargs='+', type=int, help="number of neighbors")

    args = parser.parse_args()
    num_walks_list = args.num_walks
    walk_length_list = args.walk_length
    num_dimensions_list = args.num_dimensions
    window_size_list = args.window_size
    num_neighbors_list = args.num_neighbors

    nodes, hyperedges, paperid_classid, classid_classname = get_citation_network("filePaths.txt")
    hypergraph_adj_matrix, graph_adj_matrix, index_map = get_adj_matrices(nodes, hyperedges)

    target_classes = []
    for node in nodes:
        target_classes.append(paperid_classid[node])

    for num_walks in num_walks_list:
        for walk_length in walk_length_list:
            for num_dimensions in num_dimensions_list:
                for window_size in window_size_list:
                    hypergraph_model = get_representations(nodes, hypergraph_adj_matrix, index_map,
                                                           num_walks, walk_length, num_dimensions, window_size)
                    graph_model = get_representations(nodes, graph_adj_matrix, index_map,
                                                      num_walks, walk_length, num_dimensions, window_size)

                    for num_neighbors in num_neighbors_list:
                        hypergraph_accuracy, graph_accuracy = evaluate(nodes, target_classes, hypergraph_model,
                                                                       graph_model, num_neighbors)

                        print("graph_type:hypergraph" +
                              "  num_walks:" + str(num_walks) + "  walk_length:" + str(walk_length) +
                              "  num_dimensions:" + str(num_dimensions) + "  window_size:" + str(window_size) +
                              "  num_neighbors:" + str(num_neighbors) + "  ACCURACY: " + str(hypergraph_accuracy))

                        print("graph_type:graph" +
                              "  num_walks:" + str(num_walks) + "  walk_length:" + str(walk_length) +
                              "  num_dimensions:" + str(num_dimensions) + "  window_size:" + str(window_size) +
                              "  num_neighbors:" + str(num_neighbors) + "  ACCURACY: " + str(graph_accuracy))


if __name__ == "__main__":
  sys.exit(main())
