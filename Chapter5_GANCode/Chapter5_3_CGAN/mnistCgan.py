import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from .mnistCganGenerator import build_generator
from .mnistCganDiscriminator import build_discriminator
from .mnistData import MNIST


def _check_trainable_weights_consistency(self):
    return
Model._check_trainable_weights_consistency = _check_trainable_weights_consistency


PATH = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyGANKurs")
IMAGES_PATH = os.path.join(PATH, "Chapter5_GANCode/Chapter5_3_CGAN/images")


class CGAN():
    def __init__(self):
        # Model parameters
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.z_dimension = 100
        self.num_classes = 10
        optimizer = Adam(0.0002, 0.5)

        # BUILD DISCRIMINATOR
        self.discriminator = build_discriminator(self.img_shape, self.num_classes)
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # BUILD GENERATOR
        self.generator = build_generator(self.z_dimension, self.num_classes, self.img_shape)
        noise = Input(shape=(self.z_dimension,))
        label = Input(shape=(self.num_classes,))
        img = self.generator([noise, label])
        self.discriminator.trainable = False
        d_pred = self.discriminator([img, label])
        self.combined = Model([noise, label], d_pred)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizer)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Load and rescale dataset
        mnistData = MNIST()
        x_train, y_train = mnistData.get_train_set()
        x_train = x_train / 127.5 - 1.
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Start training
        for epoch in range(epochs):
            # TRAINSET IMAGES
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            labels = y_train[idx]
            # GENERATED IMAGES
            noise = np.random.normal(0, 1, (batch_size, self.z_dimension))
            gen_imgs = self.generator.predict([noise, labels])
            # TRAIN DISCRIMINATOR
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # TRAIN GENERATOR
            noise = np.random.normal(0, 1, (batch_size, self.z_dimension))
            sampled_labels = np.random.randint(0, self.num_classes, batch_size)
            sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
            # SAVE PROGRESS
            if (epoch % sample_interval) == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                      (epoch, d_loss[0], 100 * d_loss[1], g_loss))
                self.sample_images(epoch)

    # Save sample images
    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dimension))
        sampled_labels = np.random.randint(0, self.num_classes, self.num_classes)
        sampled_labels_toc = to_categorical(sampled_labels, num_classes=self.num_classes)
        gen_imgs = self.generator.predict([noise, sampled_labels_toc])
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                axs[i, j].set_title("Digit: %d" % sampled_labels[cnt])
                cnt += 1
        fig.savefig(IMAGES_PATH + "/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=200000, batch_size=32, sample_interval=1000)
