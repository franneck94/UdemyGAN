from keras.layers import Activation  # noqa: F401
from keras.layers import BatchNormalization  # noqa: F401
from keras.layers import Conv2D  # noqa: F401
from keras.layers import Dense  # noqa: F401
from keras.layers import Dropout  # noqa: F401
from keras.layers import Flatten  # noqa: F401
from keras.layers import MaxPooling2D  # noqa: F401
from keras.models import Sequential
from keras.optimizers import Adam

from cifarData import CIFAR10


def main() -> None:
    cifar_data = CIFAR10()
    x_train, y_train = cifar_data.get_train_set()
    x_test, y_test = cifar_data.get_test_set()

    # Define the CNN
    model = Sequential()

    # Print the CNN layers
    model.summary()

    # Train the CNN
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=0.0005),
        metrics=["accuracy"],
    )
    model.fit(
        x=x_train,
        y=y_train,
        verbose=1,
        batch_size=128,
        epochs=15,
        validation_data=(x_test, y_test),
    )

    # Test the CNN
    score = model.evaluate(x=x_test, y=y_test)
    print("Test accuracy: ", score[1])


if __name__ == "__main__":
    main()
