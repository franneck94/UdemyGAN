import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model


def build_generator(z_dimension, img_shape):
    pass


if __name__ == "__main__":
    z_dimension = 100
    img_shape = (28, 28, 1)
    model = build_generator(z_dimension=z_dimension, img_shape=img_shape)
