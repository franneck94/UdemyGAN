import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import Model

from mnistData import MNIST


PATH = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyGAN")
IMAGES_PATH = os.path.join(PATH, "Chapter7_Autoencoder/images")

data = MNIST()
x_train, _ = data.get_train_set()
x_test, _ = data.get_test_set()


def build_autoencoder():
    encoding_dim = 100
    # Input Tensors
    img_shape = (28, 28, 1)
    input_img = Input(shape=img_shape)
    # Encoder Part
    x = Conv2D(filters=8, kernel_size=3, padding="same")(input_img)
    x = Activation("relu")(x)
    x = MaxPooling2D(padding="same")(x)
    x = Conv2D(filters=4, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(padding="same")(x)
    encoded = Conv2D(filters=2, kernel_size=3, padding="same")(x)
    encoded = Activation("relu")(encoded)
    # Decoder Part
    x = Conv2DTranspose(filters=4, kernel_size=3, strides=2, padding="same")(encoded)
    x = Activation("relu")(x)
    x = Conv2DTranspose(filters=4, kernel_size=3, strides=2, padding="same")(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(filters=4, kernel_size=3, strides=1, padding="same")(x)
    x = Activation("relu")(x)
    decoded = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding="same")(x)
    decoded = Activation("sigmoid")(decoded)
    # Output Tensors
    model = Model(inputs=input_img, outputs=decoded)
    model.summary()
    return model


def plot_imgs(test_imgs, decoded_imgs):
    # PLot test imgs
    plt.figure(figsize=(12, 6))
    for i in range(10):
        # Original image
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(test_imgs[i].reshape(28, 28), cmap="gray")
        # Decoded image
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.savefig(os.path.join(IMAGES_PATH, "deep_conv_autoencoder.png"))
    plt.show()


def run_autoencoder(model):
    # Training
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[]
    )
    model.fit(
        x=x_train,
        y=x_train,
        epochs=10,
        batch_size=128,
        validation_data=(x_test, x_test)
    )
    # Testing
    test_imgs = x_test[:10]
    decoded_imgs = model.predict(
        x=test_imgs
    )
    return test_imgs, decoded_imgs


if __name__ == "__main__":
    model = build_autoencoder()
    test_imgs, decoded_imgs = run_autoencoder(model)
    plot_imgs(test_imgs, decoded_imgs)
