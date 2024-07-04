from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.models import Model
from keras.models import Sequential


def build_generator(z_dimension: int, img_shape: tuple) -> Model:
    model = Sequential()
    model.add(Dense(units=7 * 7 * 128, input_dim=z_dimension))  # 6272,
    model.add(LeakyReLU(alpha=0.2))  # 6272,
    model.add(Reshape(target_shape=(7, 7, 128)))  # 7x7x128
    model.add(UpSampling2D())  # 14x14x128
    model.add(
        Conv2D(
            filters=128,
            kernel_size=5,
            strides=1,
            padding="same",
            use_bias=False,
        ),
    )  # 14x14x128
    model.add(BatchNormalization())  # 14x14x128
    model.add(LeakyReLU(alpha=0.2))  # 14x14x128
    model.add(UpSampling2D())  # 28x28x128
    model.add(
        Conv2D(
            filters=64,
            kernel_size=5,
            strides=1,
            padding="same",
            use_bias=False,
        ),
    )  # 28x28x64
    model.add(BatchNormalization())  # 28x28x64
    model.add(LeakyReLU(alpha=0.2))  # 28x28x64
    model.add(
        Conv2D(
            filters=img_shape[-1],
            kernel_size=5,
            strides=1,
            padding="same",
            use_bias=False,
        ),
    )  # 28x28x1
    model.add(Activation("tanh"))  # (-1 ,1)
    model.summary()
    noise = Input(shape=(z_dimension,))
    img = model(noise)
    return Model(inputs=noise, outputs=img)


if __name__ == "__main__":
    z_dimension = 100
    img_shape = (28, 28, 1)
    model = build_generator(z_dimension=z_dimension, img_shape=img_shape)
