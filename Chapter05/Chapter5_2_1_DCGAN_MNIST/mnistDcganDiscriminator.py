from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.models import Model
from keras.models import Sequential


def build_discriminator(img_shape: tuple) -> Model:
    model = Sequential()
    model.add(
        Conv2D(
            filters=64,
            kernel_size=5,
            strides=2,
            padding="same",
            input_shape=img_shape,
        ),
    )  # 28x28x1 => 14x14x64
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.3))
    model.add(
        Conv2D(filters=128, kernel_size=5, strides=2, padding="same"),
    )  # 14x14x62 => 7x7x128
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.3))
    model.add(Flatten())  # 7x7x128 => 6272
    model.add(Dense(units=1))
    model.add(Activation("sigmoid"))  # (0, 1)
    model.summary()
    img = Input(shape=img_shape)
    d_pred = model(img)
    return Model(inputs=img, outputs=d_pred)


if __name__ == "__main__":
    img_shape = (28, 28, 1)
    model = build_discriminator(img_shape=img_shape)
