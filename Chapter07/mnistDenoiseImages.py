import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model

from mnistData import MNIST


PATH = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyGAN")
IMAGES_PATH = os.path.join(PATH, "Chapter07/images")

mnist_data = MNIST()
x_train, _ = mnist_data.get_train_set()
x_test, _ = mnist_data.get_test_set()
x_train_noise = x_train + 0.1 * np.random.normal(size=x_train.shape)
x_test_noise = x_test + 0.1 * np.random.normal(size=x_test.shape)


def build_autoencoder() -> Model:
    encoding_dim = 100
    # Inputs
    img_shape = (28, 28, 1)
    input_img = Input(shape=img_shape)
    input_img_flatten = Flatten()(input_img)
    # Encoder
    encoded = Dense(units=256)(input_img_flatten)
    encoded = Activation("relu")(encoded)
    encoded = Dense(units=128)(encoded)
    encoded = Activation("relu")(encoded)
    encoded = Dense(units=encoding_dim)(encoded)
    encoded = Activation("relu")(encoded)
    # Decoder
    decoded = Dense(units=128)(encoded)
    decoded = Activation("relu")(decoded)
    decoded = Dense(units=256)(decoded)
    decoded = Activation("relu")(decoded)
    decoded = Dense(units=np.prod(img_shape))(decoded)
    decoded = Activation("sigmoid")(decoded)
    # Output
    output_img = Reshape(target_shape=img_shape)(decoded)
    # Model
    model = Model(inputs=input_img, outputs=output_img)
    model.summary()
    return model


def run_autoencoder(model: Model) -> tuple[np.ndarray, np.ndarray]:
    # Training
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        x=x_train_noise,
        y=x_train,
        epochs=10,
        batch_size=128,
        validation_data=(x_test_noise, x_test),
    )
    # Testing
    test_imgs = x_test_noise[:10]
    decoded_imgs = model.predict(x=test_imgs)
    return test_imgs, decoded_imgs


def plot_imgs(test_imgs: np.ndarray, decoded_imgs: np.ndarray) -> None:
    plt.figure(figsize=(12, 6))
    for i in range(10):
        _ = plt.subplot(2, 10, i + 1)
        plt.imshow(test_imgs[i].reshape(28, 28), cmap="gray")
        _ = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.savefig(os.path.join(IMAGES_PATH, "denoise_autoencoder.png"))


if __name__ == "__main__":
    model = build_autoencoder()
    test_imgs, decoded_imgs = run_autoencoder(model)
    plot_imgs(test_imgs, decoded_imgs)
