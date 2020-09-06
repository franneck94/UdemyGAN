from tensorflow.keras.layers import (Activation,
                                     BatchNormalization,
                                     Conv2D,
                                     Dense,
                                     Input,
                                     LeakyReLU,
                                     Reshape,
                                     UpSampling2D)
from tensorflow.keras.models import Model, Sequential


def build_generator(z_dimension, channels):
    model = Sequential()

    model.add(Dense(128 * 7 * 7, input_dim=z_dimension))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=5, strides=1, padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=5, strides=1, padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(channels, kernel_size=5, strides=1, padding="same", use_bias=False))
    model.add(Activation("tanh"))

    model.summary()
    noise = Input(shape=(z_dimension,))
    img = model(noise)
    return Model(inputs=noise, outputs=img)


if __name__ == "__main__":
    z_dimension = 100
    channels = 1
    g = build_generator(z_dimension, channels)
