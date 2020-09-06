from tensorflow.keras.layers import (Activation,
                                     Conv2D,
                                     Dense,
                                     Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from .mnistData2 import MNIST


mnistData = MNIST()
x_train, y_train = mnistData.get_train_set()
x_test, y_test = mnistData.get_test_set()

# Define the DNN
model = Sequential()
# Conv Block 1
model.add(Conv2D(32, (7, 7), input_shape=(28, 28, 1)))
model.add(Activation("relu"))
model.add(Conv2D(32, (5, 5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Conv Block 2
model.add(Conv2D(64, (5, 5)))
model.add(Activation("relu"))
model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# Output Layer
model.add(Dense(10))
model.add(Activation("softmax"))

# Print the DNN layers
model.summary()

# Train the DNN
lr = 0.0005
optimizer = Adam(lr=lr)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(x_train, y_train, verbose=1,
          batch_size=128, epochs=10,
          validation_data=(x_test, y_test))

# Test the DNN
score = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy: ", score[1])
