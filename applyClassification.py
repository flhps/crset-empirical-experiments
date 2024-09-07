from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
import pickle
import sys


def test_classifier(classifier, X_train, X_test, y_train, y_test):
    scores = cross_validate(classifier, X_train, y_train, cv=5)
    print("Cross-validation Score Mean:", scores["test_score"].mean())
    classifier.fit(X_train, y_train)
    print("Mean Accuracy: ", classifier.score(X_test, y_test))


def build_and_eval_classifier(X, y):
    # scikit seems to expect one-value predictions as (n_samples, ) instead of (n_samples, 1)
    if y.shape[1] == 1:
        y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)
    print("--------------------------")
    print("Stochastic Gradient Descent Classifier")
    print("--------------------------")
    clf1 = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-4))
    test_classifier(clf1, X_train, X_test, y_train, y_test)
    print("--------------------------")
    print("Linear Support Vector Classifier")
    print("--------------------------")
    clf2 = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-4))
    test_classifier(clf2, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as inp:
        d = pickle.load(inp)

    print("Training and evaluating...")
    build_and_eval_classifier(d[0], d[1])
    print("Done.")
