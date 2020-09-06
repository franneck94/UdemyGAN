from tensorflow.keras.layers import Activation, Dense, Flatten, Input, LeakyReLU
from tensorflow.keras.models import Model, Sequential


def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.summary()
    img = Input(shape=img_shape)
    d_pred = model(img)
    return Model(img, d_pred)
