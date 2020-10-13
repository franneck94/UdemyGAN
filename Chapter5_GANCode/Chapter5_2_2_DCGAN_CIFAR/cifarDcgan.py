import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from cifarData import CIFAR10
from cifarDcganDiscriminator import build_discriminator
from cifarDcganGenerator import build_generator


PATH = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyGAN")
IMAGES_PATH = os.path.join(PATH, "Chapter5_GANCode/Chapter5_2_2_DCGAN_CIFAR/images")


class DCGAN:
    def __init__(self):
        # Model parameters
        self.img_rows = 32
        self.img_cols = 32
        self.img_depth = 3
        self.img_shape = (
            self.img_rows,
            self.img_cols,
            self.img_depth
        )
        self.z_dimension = 100
        optimizer_discriminator = Adam(
            learning_rate=0.0003
        )
        optimizer_generator = Adam(
            learning_rate=0.0008
        )
        # Build Discriminator
        self.discriminator = build_discriminator(
            img_shape=self.img_shape
        )
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=optimizer_discriminator,
            metrics=["accuracy"]
        )
        # Build Generator
        self.generator = build_generator(
            z_dimension=self.z_dimension,
            img_shape=self.img_shape
        )
        z = Input(shape=(self.z_dimension,)) # Input for Generator
        img = self.generator(z) # Generator generates an image
        self.discriminator.trainable = False # Set the discriminator in non-trainable mode
        d_pred = self.discriminator(img) # Generator image as input for the discriminator
        self.combined = Model(
            inputs=z,
            outputs=d_pred
        )
        self.combined.compile(
            loss="binary_crossentropy",
            optimizer=optimizer_generator,
            metrics=[]
        )

    def train_generator(self, noise, y_real):
        g_loss = self.combined.train_on_batch(x=noise, y=y_real)
        return g_loss

    def train_discriminator(self, train_imgs, generated_imgs, y_real, y_fake, smooth=0.02):
        d_loss_real = self.discriminator.train_on_batch(x=train_imgs, y=y_real * (1.0 - smooth))
        d_loss_fake = self.discriminator.train_on_batch(x=generated_imgs, y=y_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        return d_loss

    def train(self, epochs, batch_size, sample_interval):
        # Load and rescale dataset
        cifar_data = CIFAR10()
        x_train, _ = cifar_data.get_train_set()
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
            noise = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, self.z_dimension))
            generated_imgs = self.generator(noise, training=False)
            # Training
            d_loss = self.train_discriminator(train_imgs, generated_imgs, y_real, y_fake)
            g_loss = self.train_generator(noise, y_real)
            if (epoch % sample_interval) == 0:
                print(
                    f"{epoch} - D_loss: {round(d_loss[0], 4)}"
                    f" D_acc: {round(d_loss[1], 4)}"
                    f" G_loss: {round(g_loss, 4)}"
                )
            # Save the progress
            if (epoch % sample_interval) == 0:
                self.sample_images(epoch)
        self.sample_images("final")

    def sample_images(self, epoch):
        """Save sample images

        Parameters
        ----------
        epoch : int
            Number of the current epoch
        """
        r, c = 5, 5
        noise = np.random.normal(loc=0.0, scale=1.0, size=(r * c, self.z_dimension))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :])
                axs[i, j].axis("off")
                cnt += 1
        img_name = f"{epoch}.png"
        fig.savefig(os.path.join(IMAGES_PATH, img_name))
        plt.close()


if __name__ == "__main__":
    dcgan = DCGAN()
    dcgan.train(
        epochs=50_000,
        batch_size=32,
        sample_interval=1_000
    )
