# encoding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer


def main():
    iris = load_iris()
    train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.2,
                                                                        random_state=1)
    labels_train = LabelBinarizer().fit_transform(train_target)
    model = Sequential(
        [
            Dense(5, input_dim=4),
            Activation("relu"),
            Dense(3),  # label 0 1 2
            Activation("sigmoid")
        ]
    )
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(optimizer=sgd, loss="categorical_crossentropy")
    model.fit(train_data, labels_train, nb_epoch=200, batch_size=40)
    print(model.predict_classes(test_data))
    print(test_target)


if __name__ == "__main__":
    main()
