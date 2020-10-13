import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model

from mnistData import MNIST


PATH = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyGAN")
IMAGES_PATH = os.path.join(PATH, "Chapter7_Autoencoder/images")

data = MNIST()
x_train, _ = data.get_train_set()
x_test, _ = data.get_test_set()


def build_autoencoder():
    encoding_dim = 32
    # Input Tensors
    img_shape = (28, 28, 1)
    input_img = Input(shape=img_shape)
    input_img_flatten = Flatten()(input_img)
    # Encoder Part
    x = Dense(units=256)(input_img_flatten) # 784 => 256
    x = Activation("relu")(x)
    x = Dense(units=128)(x) # 256 => 128
    x = Activation("relu")(x)
    encoded = Dense(units=encoding_dim)(x) # 128 >= 32
    encoded = Activation("relu")(encoded)
    # Decoder Part
    x = Dense(units=128)(encoded) # 32 => 128
    x = Activation("relu")(x)
    x = Dense(units=256)(x) # 128 >= 256
    x = Activation("relu")(x)
    decoded = Dense(units=np.prod(img_shape))(x) # 256 => 784
    decoded = Activation("sigmoid")(decoded)
    # Output Tensors
    output_img = Reshape(img_shape)(decoded)
    model = Model(inputs=input_img, outputs=output_img)
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
    plt.savefig(os.path.join(IMAGES_PATH, "deep_autoencoder.png"))
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
