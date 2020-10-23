from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential


def build_cnn():
    img_shape = (28, 28, 1)
    # Define the CNN
    model = Sequential()
    # Conv Block 1
    model.add(Conv2D(filters=32, kernel_size=7, input_shape=img_shape))
    model.add(Conv2D(filters=32, kernel_size=5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("relu"))
    # Conv Block 2
    model.add(Conv2D(filters=64, kernel_size=5))
    model.add(Conv2D(filters=128, kernel_size=3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("relu"))
    # Fully connected layer 1
    model.add(Flatten())
    model.add(Dense(units=512))
    model.add(Activation("relu"))
    # Fully connected layer 1
    model.add(Dense(units=256))
    model.add(Activation("relu"))
    # Output layer
    model.add(Dense(units=10))
    model.add(Activation("softmax"))
    # Print the CNN layers
    # model.summary()
    # Model object
    img = Input(shape=img_shape)
    pred = model(img)
    return Model(inputs=img, outputs=pred)
