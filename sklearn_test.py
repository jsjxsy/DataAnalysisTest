from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics


def main():
    iris = load_iris()
    print(iris)
    print(len(iris["data"]))

    train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.2,
                                                                        random_state=1)

    # 决策树
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf.fit(train_data, train_target)
    y_pred = clf.predict(test_data)

    # Verify

    print(metrics.accuracy_score(y_true=test_target, y_pred=y_pred))
    print(metrics.confusion_matrix(y_true=test_target, y_pred=y_pred))
    with open("./test/test.xlsx", "w") as f:
        tree.export_graphviz(clf, out_file=f)


if __name__ == "__main__":
    main()
