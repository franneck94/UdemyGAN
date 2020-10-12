from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model


def build_discriminator(img_shape):
    pass


if __name__ == "__main__":
    img_shape = (28, 28, 1)
    model = build_discriminator(img_shape=img_shape)
