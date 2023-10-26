import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import InputLayer
from keras.layers import Reshape
from keras.metrics import Mean
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import Optimizer

from mnistData import MNIST


PATH = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyGAN")
IMAGES_PATH = os.path.join(PATH, "Chapter7_Autoencoder/images")

data = MNIST()
x_train, y_train = data.get_train_set()
x_test, y_test = data.get_test_set()


class CVAE(Model):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder = Sequential()
        self.encoder = Sequential()
        self.build_encoder()
        self.build_decoder()

    def build_encoder(self) -> None:
        self.encoder.add(InputLayer(input_shape=(28, 28, 1)))
        self.encoder.add(
            Conv2D(filters=32, kernel_size=3, strides=2, padding="same")
        )
        self.encoder.add(Activation("relu"))
        self.encoder.add(
            Conv2D(filters=64, kernel_size=3, strides=2, padding="same")
        )
        self.encoder.add(Activation("relu"))
        self.encoder.add(Flatten())
        self.encoder.add(Dense(units=self.latent_dim + self.latent_dim))
        self.encoder.summary()

    def build_decoder(self) -> None:
        self.decoder.add(InputLayer(input_shape=(self.latent_dim,)))
        self.decoder.add(Dense(units=7 * 7 * 32))
        self.decoder.add(Activation("relu"))
        self.decoder.add(Reshape(target_shape=(7, 7, 32)))
        self.decoder.add(
            Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding="same"
            )
        )
        self.decoder.add(Activation("relu"))
        self.decoder.add(
            Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding="same"
            )
        )
        self.decoder.add(Activation("relu"))
        self.decoder.add(
            Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding="same")
        )
        self.decoder.summary()

    @tf.function
    def sample(self, eps: tf.Tensor | None = None) -> tf.Tensor:
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z: tf.Tensor, apply_sigmoid: bool = False) -> tf.Tensor:
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits


def log_normal_pdf(
    sample: tf.Tensor, mean: tf.Tensor, logvar: tf.Tensor, raxis: int = 1
) -> tf.Tensor:
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis,
    )


def compute_loss(model: CVAE, x: tf.Tensor) -> tf.Tensor:
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit, labels=x
    )
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model: CVAE, x: tf.Tensor, optimizer: Optimizer) -> None:
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def generate_and_save_images(
    model: CVAE, epoch: int, test_sample: tf.Tensor
) -> None:
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    _ = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap="gray")
        plt.axis("off")

    plt.savefig(os.path.join(IMAGES_PATH, f"image_at_epoch_{epoch:04d}.png"))


def main() -> None:
    epochs = 40
    latent_dim = 2
    optimizer = Adam(1e-4)
    model = CVAE(latent_dim)

    train_idxs: np.ndarray = np.arange(x_train.shape[0])
    np.random.shuffle(train_idxs)
    train_batches_idxs: np.ndarray = train_idxs.reshape(600, -1)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_batch_idxs in train_batches_idxs:
            train_step(model, x_train[train_batch_idxs], optimizer)
        end_time = time.time()

        loss = Mean()
        loss(compute_loss(model, x_test))
        elbo = -loss.result()
        print(
            f"Epoch: {epoch}, Test set ELBO: {elbo}, "
            f"time elapse for current epoch: {end_time - start_time}"
        )
        if epoch % 10 == 0:
            generate_and_save_images(model, epoch, x_test[:10])


if __name__ == "__main__":
    main()
