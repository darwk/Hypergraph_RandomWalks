from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


def svm_classifier(X_train, X_test, y_train, y_test, C, gamma):
    clf = SVC(decision_function_shape='ovr', C=C, gamma=gamma, random_state=0).fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test))

    return accuracy, conf_matrix


def knn_classifier(X_train, X_test, y_train, y_test, n_neighbors):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test))

    return accuracy, conf_matrix
