from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.models import Model


def build_discriminator(img_shape: tuple, num_classes: int) -> Model:
    img = Input(shape=(img_shape))
    label = Input(shape=(num_classes,))
    img_flatten = Flatten()(img)
    x = Concatenate()([img_flatten, label])
    x = Dense(units=512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate=0.25)(x)
    x = Dense(units=512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate=0.25)(x)
    x = Dense(units=1)(x)
    d_pred = Activation("sigmoid")(x)
    model = Model(inputs=[img, label], outputs=d_pred)
    model.summary()
    return model


if __name__ == "__main__":
    img_shape = (28, 28, 1)
    num_classes = 10
    model = build_discriminator(img_shape=img_shape, num_classes=num_classes)
