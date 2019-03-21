import argparse
import csv
import random
import sys
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from gensim.models import KeyedVectors
from citation_network import get_citation_network
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from utils import write_matrix_to_disk


def svm_classifier(X_train, X_test, y_train, y_test, C, gamma):
    clf = SVC(decision_function_shape='ovr', C=C, gamma=gamma, random_state=0).fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, conf_matrix, micro_f1, macro_f1, weighted_f1


def knn_classifier(X_train, X_test, y_train, y_test, n_neighbors):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, conf_matrix, micro_f1, macro_f1, weighted_f1


def classifier():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, choices=["svm", "knn"], help="Select the classifier - SVM or KNN")
    parser.add_argument("--C", nargs='+', type=float, help="SVM's C parameter")
    parser.add_argument("--gamma", nargs='+', type=float, help="SVM's gamma parameter")
    parser.add_argument("--num_neighbors", nargs='+', type=int, help="KNN's number of neighbors parameter")
    parser.add_argument("--hypergraph_model_file", type=str, help="hypergraph file model path")
    parser.add_argument("--graph_model_file", type=str, help="graph model file path")
    parser.add_argument("--test_size", nargs='+', type=float, help="proportion  of the dataset to include in the test split")
    parser.add_argument("--output", type=str, help="output folder path")

    args = parser.parse_args()
    classifier = args.classifier
    test_size_list = args.test_size
    hypergraph_model_file = args.hypergraph_model_file
    graph_model_file = args.graph_model_file
    output_folder = args.output

    nodes, hyperedges, paperid_classid, classid_classname = get_citation_network("filePaths.txt", True)

    print("Loading models")
    hypergraph_model = KeyedVectors.load_word2vec_format(hypergraph_model_file)
    graph_model = KeyedVectors.load_word2vec_format(graph_model_file)
    print("Successfully loaded models")

    rand = random.Random(0)
    for test_size in test_size_list:
        rand.shuffle(nodes)

        target_classes = []
        for node in nodes:
            target_classes.append(paperid_classid[node])

        print("Split data into training and test data with test_size - " + str(test_size))
        nodes_train, nodes_test, target_classes_train, target_classes_test = train_test_split(nodes, target_classes, test_size=test_size, stratify=target_classes)

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

        if classifier == "svm":
            C_list = args.C
            gamma_list = args.gamma

            for C in C_list:
                for gamma in gamma_list:
                    hypergraph_accuracy, hypergraph_conf_matrix, hypergraph_micro_f1, hypergraph_macro_f1, hypergraph_weighted_f1\
                        = svm_classifier(hypergraph_train, hypergraph_test, target_classes_train, target_classes_test, C, gamma)
                    graph_accuracy, graph_conf_matrix, graph_micro_f1, graph_macro_f1, graph_weighted_f1 \
                        = svm_classifier(graph_train, graph_test, target_classes_train, target_classes_test, C, gamma)
                    csvfile = open(output_folder + "/Results_svm.csv", "a")
                    csvwriter = csv.writer(csvfile)

                    row1 = ["hypergraph", "svm", str(test_size), str(C), str(gamma), str(hypergraph_accuracy),
                            str(hypergraph_micro_f1), str(hypergraph_macro_f1), str(hypergraph_weighted_f1)]
                    row2 = ["graph", "svm", str(test_size), str(C), str(gamma), str(graph_accuracy),
                            str(graph_micro_f1), str(graph_macro_f1), str(graph_weighted_f1)]

                    csvwriter.writerow(row1)
                    csvwriter.writerow(row2)

                    csvfile.close()

                    write_matrix_to_disk(output_folder + "/svm_hypergraph_conf_matrix_" + str(test_size) + "_" + str(C) + "_" + str(gamma) +
                                         ".csv", hypergraph_conf_matrix, "%i")
                    write_matrix_to_disk(output_folder + "/svm_graph_conf_matrix_" + str(test_size) + "_" + str(C) + "_" + str(gamma) + ".csv",
                                         graph_conf_matrix, "%i")

                    print("SVM Hypergraph Accuracy : test size - " + str(test_size) + ", C - " + str(C) + ", gamma - " + str(gamma) + " - " + str(hypergraph_accuracy))
                    print("SVM graph Accuracy : test size - " + str(test_size) + ", C - " + str(C) + ", gamma - " + str(gamma) + " - " + str(graph_accuracy))

        elif classifier == "knn":
            num_neighbors_list = args.num_neighbors
            for num_neighbors in num_neighbors_list:
                hypergraph_accuracy, hypergraph_conf_matrix, hypergraph_micro_f1, hypergraph_macro_f1, hypergraph_weighted_f1 \
                    = knn_classifier(hypergraph_train, hypergraph_test, target_classes_train, target_classes_test, num_neighbors)
                graph_accuracy, graph_conf_matrix, graph_micro_f1, graph_macro_f1, graph_weighted_f1 \
                    = knn_classifier(graph_train, graph_test, target_classes_train, target_classes_test, num_neighbors)

                csvfile = open(output_folder + "/Results_knn.csv", "a")
                csvwriter = csv.writer(csvfile)

                row1 = ["hypergraph", str(test_size), str(num_neighbors), str(hypergraph_accuracy),
                        str(hypergraph_micro_f1), str(hypergraph_macro_f1), str(hypergraph_weighted_f1)]
                row2 = ["graph", str(test_size), str(num_neighbors), str(graph_accuracy),
                        str(graph_micro_f1), str(graph_macro_f1), str(graph_weighted_f1)]

                csvwriter.writerow(row1)
                csvwriter.writerow(row2)

                csvfile.close()

                write_matrix_to_disk(output_folder + "/knn_hypergraph_conf_matrix_" + str(test_size) + "_" + str(num_neighbors) + ".csv",
                                     hypergraph_conf_matrix, "%i")
                write_matrix_to_disk(output_folder + "/knn_graph_conf_matrix_" + str(test_size) + "_" + str(num_neighbors) + ".csv",
                                     graph_conf_matrix, "%i")

                print("KNN Hypergraph Accuracy : test size - " + str(test_size) + ", num of neighbors - " + str(num_neighbors) + " - " + str(hypergraph_accuracy))
                print("KNN graph Accuracy : test size - " + str(test_size) + ", num of neighbors - " + str(num_neighbors) + " - " + str(graph_accuracy))


def svm_param_selection():

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, help="Proportion of data set to be split into train and test sets")
    parser.add_argument("--C", nargs='+', type=float, help="SVM's C parameter")
    parser.add_argument("--gamma", nargs='+', type=float, help="SVM's gamma parameter")
    parser.add_argument("--hypergraph_model_file", type=str, help="hypergraph file model path")
    parser.add_argument("--graph_model_file", type=str, help="graph model file path")
    parser.add_argument("--output", type=str, help="output folder path")

    args = parser.parse_args()
    test_size = args.test_size
    C_list = args.C
    gamma_list = args.gamma
    hypergraph_model_file = args.hypergraph_model_file
    graph_model_file = args.graph_model_file
    output_folder = args.output

    nodes, hyperedges, paperid_classid, classid_classname = get_citation_network("filePaths.txt", True)

    print("Loading models")
    hypergraph_model = KeyedVectors.load_word2vec_format(hypergraph_model_file)
    graph_model = KeyedVectors.load_word2vec_format(graph_model_file)
    print("Successfully loaded models")

    target_classes = []
    for node in nodes:
        target_classes.append(paperid_classid[node])

    print("Split data into training and test data with test_size - " + str(test_size))
    nodes_train, nodes_test, target_classes_train, target_classes_test = train_test_split(nodes, target_classes, test_size=test_size)

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

    param_grid = {'C': C_list, 'gamma': gamma_list}

    print("Searching hypergraph best parameters")
    hypergraph_grid_search = GridSearchCV(SVC(), param_grid)
    hypergraph_grid_search.fit(hypergraph_train, target_classes_train)

    hypergraph_best_params = hypergraph_grid_search.best_params_
    print("Got hypergraph best parameters")
    print(hypergraph_grid_search.best_params_)

    classes_pred = hypergraph_grid_search.predict(hypergraph_test)
    hypergraph_conf_matrix = confusion_matrix(target_classes_test, classes_pred)
    hypergraph_micro_f1 = f1_score(target_classes_test, classes_pred, average='micro')
    hypergraph_macro_f1 = f1_score(target_classes_test, classes_pred, average='macro')
    hypergraph_weighted_f1 = f1_score(target_classes_test, classes_pred, average='weighted')

    print("Searching graph best parameters")
    graph_grid_search = GridSearchCV(SVC(), param_grid, cv=10)
    graph_grid_search.fit(graph_train, target_classes_train)

    graph_best_params = graph_grid_search.best_params_
    print("Got graph best parameters")
    print(graph_grid_search.best_params_)

    graph_classes_pred = graph_grid_search.predict(graph_test)
    graph_conf_matrix = confusion_matrix(target_classes_test, graph_classes_pred)
    graph_micro_f1 = f1_score(target_classes_test, graph_classes_pred, average='micro')
    graph_macro_f1 = f1_score(target_classes_test, graph_classes_pred, average='macro')
    graph_weighted_f1 = f1_score(target_classes_test, graph_classes_pred, average='weighted')

    csvfile = open(output_folder + "/Results_svm_grid.csv", "a")
    csvwriter = csv.writer(csvfile)

    row1 = ["hypergraph", "svm",  str(test_size), str(hypergraph_best_params["C"]), str(hypergraph_best_params["gamma"]),
            str(hypergraph_micro_f1), str(hypergraph_macro_f1), str(hypergraph_weighted_f1)]
    row2 = ["graph", "svm", str(test_size), str(graph_best_params["C"]), str(graph_best_params["gamma"]),
            str(graph_micro_f1), str(graph_macro_f1), str(graph_weighted_f1)]

    csvwriter.writerow(row1)
    csvwriter.writerow(row2)

    csvfile.close()

    write_matrix_to_disk(
        output_folder + "/svm_hypergraph_conf_matrix_" + str(test_size) + "_" + str(hypergraph_best_params["C"]) + "_" + str(hypergraph_best_params["gamma"]) +
        ".csv", hypergraph_conf_matrix, "%i")
    write_matrix_to_disk(
        output_folder + "/svm_graph_conf_matrix_" + str(test_size) + "_" + str(graph_best_params["C"]) + "_" + str(graph_best_params["gamma"]) + ".csv",
        graph_conf_matrix, "%i")

    print("Completed")


def split_dataset(nodes, target_classes, hypergraph_model, graph_model, test_size):
    print("Split data into training and test data with test_size - " + str(test_size))
    nodes_train, nodes_test, target_classes_train, target_classes_test = train_test_split(nodes, target_classes,
                                                                                          test_size=test_size)

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
    return hypergraph_train, hypergraph_test, graph_train, graph_test, target_classes_train, target_classes_test


def knn_param_selection():

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", nargs='+', type=float, help="Proportion of data set to be split into train and test sets")
    parser.add_argument("--num_neighbors", nargs='+', type=int, help="KNN's number of neighbors parameter")
    parser.add_argument("--hypergraph_model_file", type=str, help="hypergraph file model path")
    parser.add_argument("--graph_model_file", type=str, help="graph model file path")
    parser.add_argument("--output", type=str, help="output folder path")

    args = parser.parse_args()
    test_size_list = args.test_size
    num_neighbors_list = args.num_neighbors
    hypergraph_model_file = args.hypergraph_model_file
    graph_model_file = args.graph_model_file
    output_folder = args.output

    nodes, hyperedges, paperid_classid, classid_classname = get_citation_network("filePaths.txt", True)

    print("Loading models")
    hypergraph_model = KeyedVectors.load_word2vec_format(hypergraph_model_file)
    graph_model = KeyedVectors.load_word2vec_format(graph_model_file)
    print("Successfully loaded models")

    target_classes = []
    for node in nodes:
        target_classes.append(paperid_classid[node])

    param_grid = {'n_neighbors': num_neighbors_list}

    for test_size in test_size_list:
        hypergraph_train, hypergraph_test, graph_train, graph_test, target_classes_train, target_classes_test \
            = split_dataset(nodes, target_classes,hypergraph_model, graph_model, test_size)

        print("Searching hypergraph best parameters")
        hypergraph_grid_search = GridSearchCV(KNeighborsClassifier(), param_grid)
        hypergraph_grid_search.fit(hypergraph_train, target_classes_train)
        hypergraph_best_params = hypergraph_grid_search.best_params_
        print("Got hypergraph best parameters")
        print(hypergraph_grid_search.best_params_)

        classes_pred = hypergraph_grid_search.predict(hypergraph_test)
        hypergraph_conf_matrix = confusion_matrix(target_classes_test, classes_pred)
        hypergraph_micro_f1 = f1_score(target_classes_test, classes_pred, average='micro')
        hypergraph_macro_f1 = f1_score(target_classes_test, classes_pred, average='macro')
        hypergraph_weighted_f1 = f1_score(target_classes_test, classes_pred, average='weighted')

        print("Searching graph best parameters")
        graph_grid_search = GridSearchCV(KNeighborsClassifier(), param_grid)
        graph_grid_search.fit(graph_train, target_classes_train)
        graph_best_params = graph_grid_search.best_params_
        print("Got graph best parameters")
        print(graph_grid_search.best_params_)

        graph_classes_pred = graph_grid_search.predict(graph_test)
        graph_conf_matrix = confusion_matrix(target_classes_test, graph_classes_pred)
        graph_micro_f1 = f1_score(target_classes_test, graph_classes_pred, average='micro')
        graph_macro_f1 = f1_score(target_classes_test, graph_classes_pred, average='macro')
        graph_weighted_f1 = f1_score(target_classes_test, graph_classes_pred, average='weighted')

        csvfile = open(output_folder + "/Results_knn_grid.csv", "a")
        csvwriter = csv.writer(csvfile)

        row1 = ["hypergraph", "knn",  str(test_size), str(hypergraph_best_params["n_neighbors"]),
                str(hypergraph_micro_f1), str(hypergraph_macro_f1), str(hypergraph_weighted_f1)]
        row2 = ["graph", "knn", str(test_size), str(graph_best_params["n_neighbors"]),
                str(graph_micro_f1), str(graph_macro_f1), str(graph_weighted_f1)]

        csvwriter.writerow(row1)
        csvwriter.writerow(row2)

        csvfile.close()

        write_matrix_to_disk(
            output_folder + "/knn_hypergraph_conf_matrix_" + str(test_size) + "_" + str(hypergraph_best_params["n_neighbors"]) +
            ".csv", hypergraph_conf_matrix, "%i")
        write_matrix_to_disk(
            output_folder + "/knn_graph_conf_matrix_" + str(test_size) + "_" + str(graph_best_params["n_neighbors"]) + ".csv",
            graph_conf_matrix, "%i")

    print("Completed")


if __name__ == "__main__":
    sys.exit(knn_param_selection())
