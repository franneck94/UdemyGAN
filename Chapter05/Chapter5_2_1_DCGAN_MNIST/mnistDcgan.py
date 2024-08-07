import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from packaging import version

from mnistData import MNIST
from mnistDcganDiscriminator import build_discriminator
from mnistDcganGenerator import build_generator


required_version = version.parse("2.10")
installed_version = version.parse(".".join(tf.__version__.split(".")[:2]))
if installed_version > required_version:
    from keras.optimizers.legacy import Adam
else:
    from keras.optimizers import Adam


PATH = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyGAN")
IMAGES_PATH = os.path.join(
    PATH,
    os.path.abspath("Chapter05/Chapter5_2_1_DCGAN_MNIST/images"),
)


class DCGAN:
    def __init__(self) -> None:
        # Model parameters
        self.img_rows = 28
        self.img_cols = 28
        self.img_depth = 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_depth)
        self.z_dimension = 100
        optimizer = Adam(learning_rate=0.0001)
        # Build Discriminator
        self.discriminator = build_discriminator(img_shape=self.img_shape)
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )
        # Build Generator
        self.generator = build_generator(
            z_dimension=self.z_dimension,
            img_shape=self.img_shape,
        )
        z = Input(shape=(self.z_dimension,))  # Input for Generator
        img = self.generator(z)  # Generator generates an image
        self.discriminator.trainable = (
            False  # Set the discriminator in non-trainable mode
        )
        d_pred = self.discriminator(
            img,
        )  # Generator image as input for the discriminator
        self.combined = Model(inputs=z, outputs=d_pred)
        self.combined.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=[],
        )

    def train(self, epochs: int, batch_size: int, sample_interval: int) -> None:
        # Load and resacle data
        mnist_data = MNIST()
        x_train, _ = mnist_data.get_train_set()
        x_train = (x_train / 127.5) - 1.0
        # Adverserial ground truths
        y_real = np.ones(shape=(batch_size, 1))
        y_fake = np.zeros(shape=(batch_size, 1))

        # Start the training
        for epoch in range(epochs):
            # Trainset images
            rand_idxs = np.random.randint(0, x_train.shape[0], batch_size)
            train_imgs = x_train[rand_idxs]
            # Generated images
            noise = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(batch_size, self.z_dimension),
            )
            generated_imgs = self.generator(noise, training=False)
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(train_imgs, y_real)
            d_loss_fake = self.discriminator.train_on_batch(
                generated_imgs,
                y_fake,
            )
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # Train the generator
            noise = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(batch_size, self.z_dimension),
            )
            g_loss = self.combined.train_on_batch(noise, y_real)
            # Save the progress
            if (epoch % sample_interval) == 0:
                print(
                    f"{epoch} - D_loss: {round(d_loss[0], 4)}"
                    f" D_acc: {round(d_loss[1], 4)}"
                    f" G_loss: {round(g_loss, 4)}",
                )
                self.sample_images(epoch)
        self.sample_images("final")

    def sample_images(self, epoch: str | int) -> None:
        """Save sample images

        Parameters
        ----------
        epoch : int
            Number of the current epoch
        """
        r, c = 5, 5
        noise = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(r * c, self.z_dimension),
        )
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis("off")
                cnt += 1
        img_name = f"{epoch}.png"
        fig.savefig(os.path.join(IMAGES_PATH, img_name))
        plt.close()


if __name__ == "__main__":
    dcgan = DCGAN()
    dcgan.train(epochs=50_000, batch_size=128, sample_interval=1_000)
