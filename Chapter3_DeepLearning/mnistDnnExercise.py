from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from mnistData1 import MNIST


mnistData = MNIST()
x_train, y_train = mnistData.get_train_set()
x_test, y_test = mnistData.get_test_set()

# Define the DNN
model = Sequential()
# Hidden Layer 1
model.add(Dense(512, input_shape=(784,)))
model.add(Activation("relu"))
# Hidden Layer 2
model.add(Dense(512))
model.add(Activation("relu"))
# Output Layer
model.add(Dense(10))
model.add(Activation("softmax"))

# Print the DNN layers
model.summary()

# Train the DNN
lr = 0.0001
optimizer = Adam(lr=lr)
model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])
model.fit(x_train, y_train,
          epochs=10,
          batch_size=128,
          validation_data=(x_test, y_test))

# Test the DNN
score = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy: ", score[1])
