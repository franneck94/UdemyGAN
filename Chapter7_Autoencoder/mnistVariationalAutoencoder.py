import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import keras.backend as K
from .mnistData import MNIST


# Load MNIST dataset
data = MNIST()
x_train, y_train = data.get_train_set()
x_test, y_test = data.get_test_set()

x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

latent_dim = 2
intermediate_dim = 128


# Sampling from latent distribution
def sampling(args):
    z_mean, z_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5 * z_var) * epsilon


# ENCODER
# Input
input_img = Input(shape=(784,))
# Hidden Layer before latent distribution
encoder_x = Dense(intermediate_dim, activation="relu")(input_img)
# Parameters of latent distribution
z_mean = Dense(latent_dim)(encoder_x)
z_var = Dense(latent_dim)(encoder_x)
# Sample z from latent distribution
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_var])
# encoder model
encoder = Model(input_img, [z_mean, z_var, z])
encoder.summary()

# DECODER
# Input
z = Input(shape=(latent_dim,))
# Hidden Layer after latent distribution
decoder_x = Dense(intermediate_dim, activation="relu")(z)
# Output images
output_img = Dense(784, activation="sigmoid")(decoder_x)
# decoder model
decoder = Model(z, output_img)
decoder.summary()

# GENERATOR
generator_input = Input(shape=(latent_dim,))
decoder_output = decoder(generator_input)
generator = Model(generator_input, decoder_output)


def vae_loss(input_img, output_img):
    # Binary Cross-Entropy
    xent = K.sum(K.binary_crossentropy(input_img, output_img), axis=1)
    # Kullback-Leibler Divergence
    kl = 0.5 * K.sum(K.exp(z_var) + K.square(z_mean) - 1.0 - z_var, axis=1)
    return xent + kl


# VAE
vae = Model(input_img, decoder(encoder(input_img)[2]))
vae.compile(optimizer=Adam(lr=1e-3), loss=vae_loss)
vae.summary()
vae.fit(x_train, x_train, nb_epoch=10, batch_size=128, validation_data=(x_test, x_test))

# Testing part 1
x_test_encoded = encoder.predict(x_test)[0]
plt.figure(figsize=(10, 8))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=[np.argmax(y) for y in y_test])
plt.colorbar()
plt.show()

# Testing part 2
n = 10
grid_x = np.linspace(-10, 10, n)
grid_y = np.linspace(-10, 10, n)
fig, axs = plt.subplots(n, n, figsize=(12, 12))

for i, xi in enumerate(grid_x):
    for j, yj in enumerate(grid_y):
        z_sampled = np.array([[xi, yj]])
        x_decoded = generator.predict(z_sampled)
        print(np.min(x_decoded.flatten()), np.max(x_decoded.flatten()))
        axs[i][j].imshow(x_decoded.reshape(28, 28), cmap="gray")
plt.show()
