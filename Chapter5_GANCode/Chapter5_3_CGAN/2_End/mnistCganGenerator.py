import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model


def build_generator(z_dimension, img_shape, num_classes):
    noise = Input(shape=(z_dimension,))
    label = Input(shape=(num_classes,))
    x = Concatenate()([noise, label])
    x = Dense(units=512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(units=1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(units=np.prod(img_shape))(x)
    x = Activation("tanh")(x)
    img = Reshape(target_shape=img_shape)(x)
    model = Model(
        inputs=[noise, label],
        outputs=img
    )
    model.summary()
    return model


if __name__ == "__main__":
    z_dimension = 100
    img_shape = (28, 28, 1)
    num_classes = 10
    model = build_generator(
        z_dimension=z_dimension,
        img_shape=img_shape,
        num_classes=num_classes
    )
