import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from mnistData import MNIST


PATH = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyGAN")
IMAGES_PATH = os.path.join(PATH, "Chapter7_Autoencoder/images")

data = MNIST()
x_train, y_train = data.get_train_set()
x_test, y_test = data.get_test_set()


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=(28, 28, 1)))
        self.encoder.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same"))
        self.encoder.add(tf.keras.layers.Activation("relu"))
        self.encoder.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same"))
        self.encoder.add(tf.keras.layers.Activation("relu"))
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.Dense(units=latent_dim + latent_dim))
        self.encoder.summary()

        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.InputLayer(input_shape=(latent_dim,)))
        self.decoder.add(tf.keras.layers.Dense(units=7 * 7 * 32))
        self.decoder.add(tf.keras.layers.Activation("relu"))
        self.decoder.add(tf.keras.layers.Reshape(target_shape=(7, 7, 32)))
        self.decoder.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same'))
        self.decoder.add(tf.keras.layers.Activation("relu"))
        self.decoder.add(tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same'))
        self.decoder.add(tf.keras.layers.Activation("relu"))
        self.decoder.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'))
        self.decoder.summary()

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(IMAGES_PATH, 'image_at_epoch_{:04d}.png'.format(epoch)))


if __name__ == "__main__":
    epochs = 20
    latent_dim = 2
    num_examples_to_generate = 16
    optimizer = tf.keras.optimizers.Adam(1e-4)

    random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
    model = CVAE(latent_dim)

    train_idxs = np.arange(x_train.shape[0])
    np.random.shuffle(train_idxs)
    train_batches_idxs = train_idxs.reshape(600, -1)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_batch_idxs in train_batches_idxs:
            train_step(model, x_train[train_batch_idxs], optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        loss(compute_loss(model, x_test))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, elbo, end_time - start_time))
        if epoch % 10 == 0:
            generate_and_save_images(model, epoch, x_test[:10])
