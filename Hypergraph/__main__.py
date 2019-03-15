import sys
import argparse
import csv
import numpy as np

from citation_network import get_citation_network
from classifier import svm_classifier, knn_classifier
from hypergraph import get_adj_matrices
from utils import write_matrix_to_disk
from walks import build_deepwalk_corpus
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


def get_representations(nodes, adj_matrix, index_map, num_walks, walk_length, num_dimensions, window_size):
    print("Building random walk corpus")
    random_walks = build_deepwalk_corpus(nodes, adj_matrix, index_map, num_walks, walk_length)

    print("Getting representations")
    model = Word2Vec(random_walks, size=num_dimensions, window=window_size, min_count=0, sg=1, hs=1)
    return model


def evaluate(nodes, target_classes, hypergraph_model, graph_model, svm_C, svm_gamma, knn_num_neighbors, output_folder):

    print("Split data into training and test data")
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

    print("Successfully split data into training and test data")

    svm_hypergraph_accuracy, svm_hypergraph_conf_matrix, svm_hypergraph_micro_f1, svm_hypergraph_macro_f1, svm_hypergraph_weighted_f1 \
        = svm_classifier(hypergraph_train, hypergraph_test, target_classes_train, target_classes_test, svm_C, svm_gamma)
    svm_graph_accuracy, svm_graph_conf_matrix, svm_graph_micro_f1, svm_graph_macro_f1, svm_graph_weighted_f1 \
        = svm_classifier(graph_train, graph_test, target_classes_train, target_classes_test, svm_C, svm_gamma)

    write_matrix_to_disk(output_folder + "/svm_hypergraph_conf_matrix_" + str(svm_C) + "_" + str(svm_gamma) + ".csv", svm_hypergraph_conf_matrix, "%i")
    write_matrix_to_disk(output_folder + "/svm_graph_conf_matrix_" + str(svm_C) + "_" + str(svm_gamma) + ".csv", svm_graph_conf_matrix, "%i")

    knn_hypergraph_accuracy, knn_hypergraph_conf_matrix, knn_hypergraph_micro_f1, knn_hypergraph_macro_f1, knn_hypergraph_weighted_f1 \
        = knn_classifier(hypergraph_train, hypergraph_test, target_classes_train, target_classes_test, knn_num_neighbors)
    knn_graph_accuracy, knn_graph_conf_matrix, knn_graph_micro_f1, knn_graph_macro_f1, knn_graph_weighted_f1 = \
        knn_classifier(graph_train, graph_test, target_classes_train, target_classes_test, knn_num_neighbors)

    write_matrix_to_disk(output_folder + "/knn_hypergraph_conf_matrix_" + str(knn_num_neighbors) + ".csv", knn_hypergraph_conf_matrix, "%i")
    write_matrix_to_disk(output_folder + "/knn_graph_conf_matrix_" + str(knn_num_neighbors) + ".csv", knn_graph_conf_matrix, "%i")

    return svm_hypergraph_accuracy, svm_graph_accuracy, knn_hypergraph_accuracy, knn_graph_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_walks", nargs='+', type=int, help="number of random walks per node")
    parser.add_argument("--walk_length", nargs='+', type=int, help="length of the walk")
    parser.add_argument("--num_dimensions", nargs='+', type=int, help="number of dimensions")
    parser.add_argument("--window_size", nargs='+', type=int, help="window size")
    parser.add_argument("--use_cc", type=bool, help="whether to use connected component or not")
    parser.add_argument("--output", type=str, help="output folder path")

    args = parser.parse_args()
    num_walks_list = args.num_walks
    walk_length_list = args.walk_length
    num_dimensions_list = args.num_dimensions
    window_size_list = args.window_size
    use_cc = args.use_cc
    output_folder = args.output

    print("Getting network ")
    nodes, hyperedges, paperid_classid, classid_classname = get_citation_network("filePaths.txt", use_cc)

    hypergraph_adj_matrix, graph_adj_matrix, index_map = get_adj_matrices(nodes, hyperedges)

    C = 1.0
    gamma = 0.1
    num_neighbors = 5

    for num_walks in num_walks_list:
        for walk_length in walk_length_list:
            for num_dimensions in num_dimensions_list:
                for window_size in window_size_list:
                    hypergraph_model = get_representations(nodes, hypergraph_adj_matrix, index_map,
                                                           num_walks, walk_length, num_dimensions, window_size)

                    hypergraph_model.wv.save_word2vec_format(output_folder +"/hypergraph_model_" + str(num_walks) + "_"
                                                             + str(walk_length) + "_" + str(num_dimensions) + "_" + str(window_size) + ".txt")
                    print("Successfully got hypergraph model")

                    graph_model = get_representations(nodes, graph_adj_matrix, index_map,
                                                      num_walks, walk_length, num_dimensions, window_size)
                    graph_model.wv.save_word2vec_format(output_folder +"/graph_model_" + str(num_walks) + "_"
                                                             + str(walk_length) + "_" + str(num_dimensions) + "_" + str(window_size) + ".txt")
                    print("Successfully got graph model")

                    target_classes = []
                    for node in nodes:
                        target_classes.append(paperid_classid[node])

                    print("Evaluating models : ")
                    print("SVM Classifier Parameters : C - " + str(C) + " gamma - " + str(gamma))
                    print("KNN Classifier Parameter : num_neighbors - " + str(num_neighbors))

                    svm_hypergraph_accuracy, svm_graph_accuracy, knn_hypergraph_accuracy, knn_graph_accuracy = \
                        evaluate(nodes, target_classes, hypergraph_model, graph_model, C, gamma, num_neighbors, output_folder)

                    csvfile = open("Results.csv", "a")
                    csvwriter = csv.writer(csvfile)

                    row1 = ["hypergraph", "svm", str(num_walks), str(walk_length), str(num_dimensions),
                            str(window_size), str(num_neighbors), str(svm_hypergraph_accuracy)]

                    row2 = ["graph", "svm", str(num_walks), str(walk_length), str(num_dimensions),
                            str(window_size), str(num_neighbors), str(svm_graph_accuracy)]

                    row3 = ["hypergraph", "knn", str(num_walks), str(walk_length), str(num_dimensions),
                            str(window_size), str(num_neighbors), str(knn_hypergraph_accuracy)]

                    row4 = ["graph", "knn", str(num_walks), str(walk_length), str(num_dimensions),
                            str(window_size), str(num_neighbors), str(knn_graph_accuracy)]

                    csvwriter.writerow(row1)
                    csvwriter.writerow(row2)
                    csvwriter.writerow(row3)
                    csvwriter.writerow(row4)

                    csvfile.close()

                    print("graph_type:hypergraph, classifier:svm" +
                          "  num_walks:" + str(num_walks) + "  walk_length:" + str(walk_length) +
                          "  num_dimensions:" + str(num_dimensions) + "  window_size:" + str(window_size) +
                          "  num_neighbors:" + str(num_neighbors) + "  ACCURACY: " + str(svm_hypergraph_accuracy))

                    print("graph_type:graph, classifier:svm" +
                          "  num_walks:" + str(num_walks) + "  walk_length:" + str(walk_length) +
                          "  num_dimensions:" + str(num_dimensions) + "  window_size:" + str(window_size) +
                          "  num_neighbors:" + str(num_neighbors) + "  ACCURACY: " + str(svm_graph_accuracy))

                    print("graph_type:hypergraph, classifier:knn" +
                          "  num_walks:" + str(num_walks) + "  walk_length:" + str(walk_length) +
                          "  num_dimensions:" + str(num_dimensions) + "  window_size:" + str(window_size) +
                          "  num_neighbors:" + str(num_neighbors) + "  ACCURACY: " + str(knn_hypergraph_accuracy))

                    print("graph_type:graph, Classifier:knn" +
                          "  num_walks:" + str(num_walks) + "  walk_length:" + str(walk_length) +
                          "  num_dimensions:" + str(num_dimensions) + "  window_size:" + str(window_size) +
                          "  num_neighbors:" + str(num_neighbors) + "  ACCURACY: " + str(knn_graph_accuracy))


if __name__ == "__main__":
  sys.exit(main())
