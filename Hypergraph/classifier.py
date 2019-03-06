from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from utils import write_matrix_to_disk, load_matrix
from gensim.models import KeyedVectors

import numpy as np


def svm_classifier(X_train, X_test, y_train, y_test, filepath):
    clf = SVC(decision_function_shape='ovr', C=10000000.0, gamma='auto', random_state=0).fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test))

    print("accuracy ", accuracy)
    write_matrix_to_disk(filepath, conf_matrix, "%i")

    return accuracy, conf_matrix


def knn_classifier(X_train, X_test, y_train, y_test, filepath):
    clf = KNeighborsClassifier(n_neighbors=1000).fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test))

    print("accuracy ", accuracy)
    write_matrix_to_disk(filepath, conf_matrix, "%i")

    return accuracy, conf_matrix


hypergraph_model = KeyedVectors.load_word2vec_format("OutputFiles/hypergraph_node_embeddings.txt")
graph_model = KeyedVectors.load_word2vec_format("OutputFiles/graph_node_embeddings.txt")

paperid_classid = load_matrix("OutputFiles/paperid_classid.txt", " ", dtype="int")

nodes = paperid_classid[:, 0]
classes = paperid_classid[:, 1]

nodes_train, nodes_test, classes_train, classes_test = train_test_split(nodes, classes, random_state=1234)

hypergraph_train = []
graph_train = []

for node in nodes_train:
    hypergraph_train.append(hypergraph_model[str(node)])
    graph_train.append(graph_model[str(node)])

hypergraph_train = np.array(hypergraph_train)
graph_train = np.array(graph_train)

hypergraph_test = []
graph_test = []

for node in nodes_test:
    hypergraph_test.append(hypergraph_model[str(node)])
    graph_test.append(graph_model[str(node)])

hypergraph_test = np.array(hypergraph_test)
graph_test = np.array(graph_test)

write_matrix_to_disk("OutputFiles/hypergraph_train.csv", hypergraph_train, "%.5e")
print("hypergraph_train ", hypergraph_train.shape)
print("hypergraph_test ", hypergraph_test.shape)

write_matrix_to_disk("OutputFiles/graph_train.csv", graph_train, "%.5e")
print("graph_train ", graph_train.shape)
print("graph_test ", graph_test.shape)

#svm_classifier(hypergraph_train, hypergraph_test, classes_train, classes_test, "OutputFiles/svm_hypergraph_confusion_matrix.csv")
#svm_classifier(graph_train, graph_test, classes_train, classes_test, "OutputFiles/svm_graph_confusion_matrix.csv")

knn_classifier(hypergraph_train, hypergraph_test, classes_train, classes_test, "OutputFiles/hypergraph_confusion_matrix.csv")
knn_classifier(graph_train, graph_test, classes_train, classes_test, "OutputFiles/graph_confusion_matrix.csv")


