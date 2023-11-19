from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

from cifarData import CIFAR10


def main():
    cifar_data = CIFAR10()
    x_train, y_train = cifar_data.get_train_set()
    x_test, y_test = cifar_data.get_test_set()

    # Define the CNN
    model = Sequential()
    # Conv Block 1
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            input_shape=(32, 32, 3),
            padding="same",
        )
    )  # 16x16x32
    model.add(
        Conv2D(filters=64, kernel_size=(3, 3), padding="same")
    )  # 16x16x64
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 16x16x64
    model.add(Activation("relu"))  # 16x16x64
    model.add(BatchNormalization())  # 16x16x64
    # Conv Block 2
    model.add(
        Conv2D(filters=128, kernel_size=(3, 3), padding="same")
    )  # 16x16x128
    model.add(
        Conv2D(filters=128, kernel_size=(3, 3), padding="same")
    )  # 16x16x256
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 8x8x256
    model.add(Activation("relu"))  # 8x8x256
    model.add(BatchNormalization())  # 8x8x256
    # Conv Block 3
    model.add(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same")
    )  # 8x8x512
    model.add(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same")
    )  # 8x8x1024
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 4x4x1024
    model.add(Activation("relu"))  # 4x4x1024
    model.add(BatchNormalization())  # 4x4x1024
    # Fully connected layer 1
    model.add(Flatten())  # 4096,
    model.add(Dense(units=256))  # 256,
    model.add(Activation("relu"))  # 256,
    model.add(BatchNormalization())  # 256,
    # Output layer
    model.add(Dense(units=10))
    model.add(Activation("softmax"))

    # Print the CNN layers
    model.summary()

    # Train the CNN
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=0.0007),
        metrics=["accuracy"],
    )
    model.fit(
        x=x_train,
        y=y_train,
        verbose=1,
        batch_size=128,
        epochs=10,
        validation_data=(x_test, y_test),
    )

    # Test the CNN
    score = model.evaluate(x=x_test, y=y_test)
    print("Test accuracy: ", score[1])


if __name__ == "__main__":
    main()
