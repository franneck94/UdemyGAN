import matplotlib.pyplot as plt

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from mnistData import *

# Load MNIST dataset
data = MNIST()
x_train, _ = data.get_train_set()
x_test, _ = data.get_test_set()

# Encoded dimension
encoding_dim = 32

# Input Tensors
input_img = Input(shape=(28,28,1,))
input_img_flatten = Flatten()(input_img)

#Encoder Part
encoded = Dense(256, activation="relu")(input_img_flatten) # 784 => 256
encoded = Dense(128, activation="relu")(encoded) # 256 => 128
encoded = Dense(encoding_dim, activation="relu")(encoded) # 128 >= 32

# Decoder Part
decoded = Dense(128, activation="relu")(encoded) # 32 => 128
decoded = Dense(256, activation="relu")(decoded) # 128 >= 256
decoded = Dense(784, activation="sigmoid")(decoded) # 256 => 784

# Output Tensors
output_img = Reshape((28,28,1,))(decoded)
autoencoder = Model(inputs=input_img, outputs=output_img)

# Training
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(x_train, x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test))

# Testing
test_images = x_test[:10]
decoded_imgs = autoencoder.predict(test_images)

# PLot test images
plt.figure(figsize=(12,6))
for i in range(10):
    # Original image
    ax = plt.subplot(2 , 10, i+1)
    plt.imshow(test_images[i].reshape(28,28), cmap="gray")
    # Decoded image
    ax = plt.subplot(2 , 10, i+1+10)
    plt.imshow(decoded_imgs[i].reshape(28,28), cmap="gray")
plt.show()
