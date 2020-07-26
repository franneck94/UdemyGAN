from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

def build_generator(z_dimension, num_classes, img_shape):
    noise = Input(shape=(z_dimension,))
    label = Input(shape=(num_classes,))

    x = Concatenate()([noise, label])

    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(np.prod(img_shape))(x)
    x = Activation("tanh")(x)
    img = Reshape(img_shape)(x)

    model = Model(inputs=[noise, label], outputs=img)
    model.summary()
    return model
