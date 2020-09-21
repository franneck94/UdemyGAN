from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model


def build_discriminator(img_shape, num_classes):
    img = Input(shape=img_shape)
    label = Input(shape=(num_classes,))

    img_flatten = Flatten()(img)
    x = Concatenate()([img_flatten, label])

    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Dense(1)(x)
    d_pred = Activation("sigmoid")(x)

    model = Model(inputs=[img, label], outputs=d_pred)
    model.summary()
    return model
