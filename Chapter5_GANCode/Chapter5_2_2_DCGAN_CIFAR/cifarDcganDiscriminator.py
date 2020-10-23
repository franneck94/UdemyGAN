from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential


def build_discriminator(img_shape):
    model = Sequential() # 32x32x3
    model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', input_shape=(img_shape)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(filters=64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(filters=128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(filters=256, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(rate=0.3))
    model.add(Flatten())
    model.add(Dense(units=1))
    model.add(Activation("sigmoid"))
    model.summary()
    img = Input(shape=img_shape)
    d_pred = model(img)
    return Model(inputs=img, outputs=d_pred)
