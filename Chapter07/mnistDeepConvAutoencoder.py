import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Input
from keras.models import Model

from mnistData import MNIST


PATH = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyGAN")
IMAGES_PATH = os.path.join(PATH, "Chapter07/images")

mnist_data = MNIST()
x_train, _ = mnist_data.get_train_set()
x_test, _ = mnist_data.get_test_set()


def build_autoencoder() -> Model:
    # encoding_dim = 8
    # Inputs
    img_shape = (28, 28, 1)
    input_img = Input(shape=img_shape)
    # Encoder
    encoded = Conv2D(filters=8, kernel_size=3, strides=2, padding="same")(
        input_img
    )
    encoded = Activation("relu")(encoded)
    encoded = Conv2D(filters=4, kernel_size=3, strides=2, padding="same")(
        encoded
    )
    encoded = Activation("relu")(encoded)
    encoded = Conv2D(filters=1, kernel_size=3, strides=1, padding="same")(
        encoded
    )
    encoded = Activation("relu")(encoded)
    # Decoder
    decoded = Conv2DTranspose(
        filters=4, kernel_size=3, strides=2, padding="same"
    )(encoded)
    decoded = Activation("relu")(decoded)
    decoded = Conv2DTranspose(
        filters=4, kernel_size=3, strides=2, padding="same"
    )(decoded)
    decoded = Activation("relu")(decoded)
    decoded = Conv2DTranspose(
        filters=4, kernel_size=3, strides=1, padding="same"
    )(decoded)
    decoded = Activation("relu")(decoded)
    decoded = Conv2DTranspose(
        filters=1, kernel_size=3, strides=1, padding="same"
    )(decoded)
    output_img = Activation("sigmoid")(decoded)
    # Model
    model = Model(inputs=input_img, outputs=output_img)
    model.summary()
    return model


def run_autoencoder(model: Model) -> tuple[np.ndarray, np.ndarray]:
    # Training
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        x=x_train,
        y=x_train,
        epochs=20,
        batch_size=128,
        validation_data=(x_test, x_test),
    )
    # Testing
    test_imgs = x_test[:10]
    decoded_imgs = model.predict(x=test_imgs)
    return test_imgs, decoded_imgs


def plot_imgs(test_imgs: np.ndarray, decoded_imgs: np.ndarray) -> None:
    plt.figure(figsize=(12, 6))
    for i in range(10):
        _ = plt.subplot(2, 10, i + 1)
        plt.imshow(test_imgs[i].reshape(28, 28), cmap="gray")
        _ = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.savefig(os.path.join(IMAGES_PATH, "deep_conv_autoencoder.png"))


if __name__ == "__main__":
    model = build_autoencoder()
    test_imgs, decoded_imgs = run_autoencoder(model)
    plot_imgs(test_imgs, decoded_imgs)
