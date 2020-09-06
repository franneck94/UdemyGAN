from tensorflow.keras.layers import (Activation,
                                     BatchNormalization,
                                     Conv2D,
                                     Dense,
                                     Dropout,
                                     Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from .cifarData import CIFAR10


cifar_data = CIFAR10()
x_train, y_train = cifar_data.get_train_set()
x_test, y_test = cifar_data.get_test_set()

# Define the CNN
model = Sequential()
# Conv Block 1
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding="same"))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))  # 16x16
model.add(Activation("relu"))
model.add(BatchNormalization())
# Conv Block 2
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))  # 8x8
model.add(Activation("relu"))
model.add(BatchNormalization())
# Conv Block 3
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Conv2D(1024, (3, 3), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))  # 4x4
model.add(Activation("relu"))
model.add(BatchNormalization())
# Fully connected layer 1
model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.25))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Dropout(0.25))
model.add(Activation("relu"))
model.add(BatchNormalization())
# Output layer
model.add(Dense(10))
model.add(Activation("softmax"))

# Print the CNN layers
model.summary()

# Train the CNN
lr = 0.0005
optimizer = Adam(lr=lr)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(x_train, y_train, verbose=1,
          batch_size=128, nb_epoch=15,
          validation_data=(x_test, y_test))

# Test the CNN
score = model.evaluate(x_test, y_test)
print("Test accuracy: ", score[1])
