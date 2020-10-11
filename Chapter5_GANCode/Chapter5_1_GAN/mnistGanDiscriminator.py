from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential


def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape)) # 28x28x1 => 784
    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=1))
    model.add(Activation("sigmoid")) # (0, 1)
    model.summary()
    img = Input(shape=img_shape)
    d_pred = model(img)
    return Model(inputs=img, outputs=d_pred)


if __name__ == "__main__":
    img_shape = (28, 28, 1)
    model = build_discriminator(img_shape=img_shape)
