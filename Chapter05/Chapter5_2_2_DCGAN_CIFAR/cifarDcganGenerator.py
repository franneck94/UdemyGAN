from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.models import Model
from keras.models import Sequential


def build_generator(z_dimension: int, img_shape: tuple) -> Model:
    model = Sequential()
    model.add(Dense(units=2 * 2 * 512, input_shape=(z_dimension,)))
    model.add(Reshape(target_shape=(2, 2, 512)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(
        Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding="same")
    )
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(
        Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same")
    )
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(
        Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same")
    )
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(
        Conv2DTranspose(
            filters=img_shape[-1], kernel_size=3, strides=2, padding="same"
        )
    )
    model.add(Activation("tanh"))
    model.summary()
    noise = Input(shape=(z_dimension,))
    img = model(noise)
    return Model(inputs=noise, outputs=img)
