import numpy as np
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.models import Model


def build_generator(
    z_dimension: int,
    img_shape: tuple,
    num_classes: int,
) -> Model:
    noise = Input(shape=(z_dimension,))
    label = Input(shape=(num_classes,))
    x = Concatenate()([noise, label])
    x = Dense(units=512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(units=1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(np.prod(img_shape))(x)
    x = Activation("tanh")(x)
    img = Reshape(target_shape=img_shape)(x)
    model = Model(inputs=[noise, label], outputs=img)
    model.summary()
    return model


if __name__ == "__main__":
    z_dimension = 100
    img_shape = (28, 28, 1)
    num_classes = 10
    model = build_generator(
        z_dimension=z_dimension,
        img_shape=img_shape,
        num_classes=num_classes,
    )
