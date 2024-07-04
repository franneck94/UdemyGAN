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
    model = Sequential()  # 32x32x3
    model.add(
        Conv2D(
            filters=32,
            kernel_size=3,
            strides=2,
            padding="same",
            input_shape=(img_shape),
        ),
    )
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(filters=64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(filters=128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(filters=256, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(rate=0.3))
    model.add(Flatten())
    model.add(Dense(units=1))
    model.add(Activation("sigmoid"))
    model.summary()
    img = Input(shape=img_shape)
    d_pred = model(img)
    return Model(inputs=img, outputs=d_pred)
