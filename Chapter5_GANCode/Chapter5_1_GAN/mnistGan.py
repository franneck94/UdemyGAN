import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from mnistData import MNIST
from mnistGanDiscriminator import build_discriminator
from mnistGanGenerator import build_generator


PATH = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyGAN")
IMAGES_PATH = os.path.join(PATH, "Chapter5_GANCode/Chapter5_1_GAN/images")


class GAN:
    def __init__(self):
        # Model parameters
        self.img_rows = 28
        self.img_cols = 28
        self.img_depth = 1
        self.img_shape = (
            self.img_rows,
            self.img_cols,
            self.img_depth
        )
        self.z_dimension = 100
        optimizer = Adam(
            learning_rate=0.0002,
            beta_1=0.5
        )
        # Build Discriminator
        self.discriminator = build_discriminator(
            img_shape=self.img_shape
        )
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
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
            optimizer=optimizer,
            metrics=[]
        )

    def train(self, epochs, batch_size, sample_interval):
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
            noise = np.random.normal(0, 1, (batch_size, self.z_dimension))
            generated_imgs = self.generator(noise, training=False)
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(train_imgs, y_real)
            d_loss_fake = self.discriminator.train_on_batch(generated_imgs, y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, self.z_dimension))
            g_loss = self.combined.train_on_batch(noise, y_real)
            # Save the progress
            if (epoch % sample_interval) == 0:
                print(
                    f"{epoch} - D_loss: {round(d_loss[0], 4)}"
                    f" D_acc: {round(d_loss[1], 4)}"
                    f" G_loss: {round(g_loss, 4)}"
                )
                self.sample_images(epoch)

    # Save sample images
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dimension))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(IMAGES_PATH + "/%d.png" % epoch)
        plt.close()


if __name__ == "__main__":
    gan = GAN()
    gan.train(
        epochs=10_000,
        batch_size=32,
        sample_interval=1_000
    )
