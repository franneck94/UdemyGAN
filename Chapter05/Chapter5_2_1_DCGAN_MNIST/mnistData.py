import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical


class MNIST:
    def __init__(self) -> None:
        (_x_train, _y_train), (_x_test, _y_test) = mnist.load_data()
        # reshape
        self.x_train: np.ndarray = _x_train.reshape(
            _x_train.shape[0], 28, 28, 1
        )
        self.x_test: np.ndarray = _x_test.reshape(_x_test.shape[0], 28, 28, 1)
        # convert from int to float
        self.x_train = self.x_train.astype("float32")
        self.x_test = self.x_test.astype("float32")
        # rescale values
        # self.x_train = (self.x_train / (255.0 / 2.0)) - 1.0
        # self.x_test = (self.x_test / (255.0 / 2.0)) - 1.0
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        # Create one hot array
        self.y_train: np.ndarray = to_categorical(_y_train, 10)
        self.y_test: np.ndarray = to_categorical(_y_test, 10)

    def get_train_set(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.x_train, self.y_train)

    def get_test_set(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.x_test, self.y_test)
