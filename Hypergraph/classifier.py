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
from sklearn.linear_model import LogisticRegression

from utils import write_matrix_to_disk


def split_dataset(nodes, target_classes, hypergraph_model, graph_model, test_size):
    print("Split data into training and test data with test_size - " + str(test_size))
    nodes_train, nodes_test, target_classes_train, target_classes_test = \
        train_test_split(nodes, target_classes, test_size=test_size, stratify=target_classes)

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


def classifier():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["aminer", "citeseer", "cora"],
                        help="Dataset to be used - aminer or citeseer or cora")

    parser.add_argument("--use_cc", type=str,
                        help="whether to use connected component or not")

    parser.add_argument("--classifier", type=str, choices=["svm", "knn", "logit"],
                        help="Select the classifier - SVM or KNN or Logit")

    parser.add_argument("--C", nargs='+', type=float,
                        help="SVM's C parameter")

    parser.add_argument("--gamma", nargs='+', type=float,
                        help="SVM's gamma parameter")

    parser.add_argument("--n_neighbors", nargs='+', type=int,
                        help="KNN's number of neighbors parameter")

    parser.add_argument("--hypergraph_model_file", type=str,
                        help="hypergraph file model path")

    parser.add_argument("--graph_model_file", type=str,
                        help="graph model file path")

    parser.add_argument("--test_size", nargs='+', type=float,
                        help="proportion  of the dataset to include in the test split")

    parser.add_argument("--repetition_count", type=int,
                        help="number of repetitions to perform")

    parser.add_argument("--output", type=str,
                        help="output folder path")

    args = parser.parse_args()

    dataset = args.dataset
    use_cc = args.use_cc
    classifier = args.classifier
    test_size_list = args.test_size
    hypergraph_model_file = args.hypergraph_model_file
    graph_model_file = args.graph_model_file
    repetition_count = args.repetition_count
    output_folder = args.output

    nodes, hyperedges, paperid_classid, classid_classname = get_citation_network(dataset, use_cc)

    print("Loading models")
    hypergraph_model = KeyedVectors.load_word2vec_format(hypergraph_model_file)
    graph_model = KeyedVectors.load_word2vec_format(graph_model_file)
    print("Successfully loaded models")

    rand = random.Random(0)
    for test_size in test_size_list:
        hypergraph_micro_f1 = 0
        hypergraph_macro_f1 = 0
        hypergraph_weighted_f1 = 0

        graph_micro_f1 = 0
        graph_macro_f1 = 0
        graph_weighted_f1 = 0

        for i in range(repetition_count):
            rand.shuffle(nodes)

            target_classes = []
            for node in nodes:
                target_classes.append(paperid_classid[node])

            hypergraph_train, hypergraph_test, graph_train, graph_test, target_classes_train, target_classes_test \
                = split_dataset(nodes, target_classes, hypergraph_model, graph_model, test_size)

            if classifier == "svm":
                C_list = args.C
                gamma_list = args.gamma

                param_grid = {'C': C_list, 'gamma': gamma_list, 'kernel': ['linear', 'rbf']}

                hypergraph_grid_search = GridSearchCV(SVC(), param_grid, cv=5)
                graph_grid_search = GridSearchCV(SVC(), param_grid, cv=5)

            elif classifier == "knn":
                n_neighbors_list = args.n_neighbors

                param_grid = {'n_neighbors': n_neighbors_list}

                hypergraph_grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
                graph_grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)

            elif classifier == "logit":
                param_grid = {'multi_class': ['ovr']}

                hypergraph_grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
                graph_grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

            print("Searching hypergraph best parameters")
            hypergraph_grid_search.fit(hypergraph_train, target_classes_train)
            hypergraph_best_params = hypergraph_grid_search.best_params_
            print("Got hypergraph best parameters")
            print(hypergraph_best_params)

            classes_pred = hypergraph_grid_search.predict(hypergraph_test)
            hypergraph_conf_matrix = confusion_matrix(target_classes_test, classes_pred)
            hypergraph_micro_f1 += f1_score(target_classes_test, classes_pred, average='micro')
            hypergraph_macro_f1 += f1_score(target_classes_test, classes_pred, average='macro')
            hypergraph_weighted_f1 += f1_score(target_classes_test, classes_pred, average='weighted')

            print("Searching graph best parameters")
            graph_grid_search.fit(graph_train, target_classes_train)
            graph_best_params = graph_grid_search.best_params_
            print("Got graph best parameters")
            print(graph_best_params)

            graph_classes_pred = graph_grid_search.predict(graph_test)
            graph_conf_matrix = confusion_matrix(target_classes_test, graph_classes_pred)
            graph_micro_f1 += f1_score(target_classes_test, graph_classes_pred, average='micro')
            graph_macro_f1 += f1_score(target_classes_test, graph_classes_pred, average='macro')
            graph_weighted_f1 += f1_score(target_classes_test, graph_classes_pred, average='weighted')

            string1 = ""
            for key in hypergraph_best_params:
                string1 += "_" + str(hypergraph_best_params[key])
            write_matrix_to_disk(output_folder + "/confusion_matrices/" + classifier + "_hypergraph_conf_matrix_" +
                                 str(test_size) + string1 + "_" + str(i) + ".csv", hypergraph_conf_matrix, "%i")

            string2 = ""
            for key in graph_best_params:
                string2 += "_" + str(graph_best_params[key])
            write_matrix_to_disk(output_folder + "/confusion_matrices/" + classifier + "_graph_conf_matrix_" +
                                 str(test_size) + string2 + "_" + str(i) + ".csv", graph_conf_matrix, "%i")

        hypergraph_micro_f1 = (hypergraph_micro_f1/repetition_count)
        hypergraph_macro_f1 = (hypergraph_macro_f1/repetition_count)
        hypergraph_weighted_f1 = (hypergraph_weighted_f1/repetition_count)

        graph_micro_f1 = (graph_micro_f1/repetition_count)
        graph_macro_f1 = (graph_macro_f1/repetition_count)
        graph_weighted_f1 = (graph_weighted_f1/repetition_count)

        print("Writing the results to output file")
        csvfile = open(output_folder + "/Results_" + classifier + "_grid.csv", "a")
        csvwriter = csv.writer(csvfile)

        hypergraph_results = ["hypergraph", classifier, str(test_size)]
        for keys in hypergraph_best_params:
            hypergraph_results.append(str(hypergraph_best_params[keys]))
        hypergraph_results += [str(hypergraph_micro_f1), str(hypergraph_macro_f1), str(hypergraph_weighted_f1)]

        graph_results = ["graph", classifier, str(test_size)]
        for key in graph_best_params:
            graph_results.append(str(hypergraph_best_params[key]))
        graph_results += [str(graph_micro_f1), str(graph_macro_f1), str(graph_weighted_f1)]

        csvwriter.writerow(hypergraph_results)
        csvwriter.writerow(graph_results)
        csvfile.close()
        print("Successfully wrote results to output file")


if __name__ == "__main__":
    sys.exit(classifier())
