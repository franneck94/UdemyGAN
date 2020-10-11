from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from cifarData import CIFAR10


cifar_data = CIFAR10()
x_train, y_train = cifar_data.get_train_set()
x_test, y_test = cifar_data.get_test_set()

# Define the CNN
model = Sequential()
# TODO

# Print the CNN layers
model.summary()

# Train the CNN
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(lr=0.0005),
    metrics=["accuracy"]
)
model.fit(
    x=x_train,
    y=y_train,
    verbose=1,
    batch_size=128,
    epochs=15,
    validation_data=(x_test, y_test)
)

# Test the CNN
score = model.evaluate(
    x=x_test,
    y=y_test
)
print("Test accuracy: ", score[1])
