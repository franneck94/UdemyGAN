from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU
)
from tensorflow.keras.models import Model, Sequential


def build_discriminator(img_shape):
    model = Sequential()  # 28x28

    model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=img_shape, padding="same"))  # 14x14
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.summary()
    img = Input(shape=img_shape)
    d_pred = model(img)
    return Model(inputs=img, outputs=d_pred)


if __name__ == "__main__":
    img_shape = (28, 28, 1)
    d = build_discriminator(img_shape)
