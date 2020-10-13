import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from mnistCnn import build_cnn
from mnistData import MNIST
from plotting import plot_img


PATH = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyGAN")
MODELS_PATH = os.path.join(PATH, "models")
CNN_MODEL_PATH = os.path.join(MODELS_PATH, "mnist_cnn.h5")

mnistData = MNIST()
x_train, y_train = mnistData.get_train_set()
x_test, y_test = mnistData.get_test_set()


def adversarial_noise(model, image, label):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = tf.reshape(model(image, training=False), (10,))
        loss = loss_object(label, prediction)
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, image)
    # Get the sign of the gradients to create the noise
    signed_grad = tf.sign(gradient)
    return signed_grad


def train_and_save_model():
    model = build_cnn()
    optimizer = Adam(
        learning_rate=0.0005
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    model.fit(
        x=x_train,
        y=y_train,
        verbose=1,
        batch_size=256,
        epochs=10,
        validation_data=(x_test, y_test)
    )
    model.save_weights(filepath=CNN_MODEL_PATH)


def load_model():
    model = build_cnn()
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    model.load_weights(filepath=CNN_MODEL_PATH)
    return model


def untargeted_attack(model):
    sample_idx = np.random.randint(low=0, high=x_test.shape[0])
    image = np.array([x_test[sample_idx]])
    true_label = y_test[sample_idx]
    true_label_idx = np.argmax(true_label)
    y_pred = model.predict(image)[0]
    print("Right class: ", true_label_idx)
    print("Prob. right class: ", y_pred[true_label_idx])

    eps = 0.001
    image_adv = tf.convert_to_tensor(image, dtype=tf.float32)
    noise = tf.convert_to_tensor(np.zeros_like(image), dtype=tf.float32)
    target_label_idx = 9
    target_label = tf.one_hot(target_label_idx, 10)

    while (np.argmax(y_pred) != target_label_idx):
        # image_adv = image_adv + eps * noise
        noise = adversarial_noise(model, image_adv, target_label)
        if np.sum(noise) == 0.0:
            break
        image_adv = image_adv - eps * noise
        image_adv = tf.clip_by_value(image_adv, 0, 1)
        y_pred = model.predict(image_adv)[0]
        print("Prob. right class: ", y_pred[true_label_idx])
        print("Prob. target class: ", y_pred[target_label_idx], "\n")

    plot_img(image_adv.numpy(), cmap="gray")
    plot_img(noise.numpy(), cmap="gray")


if __name__ == "__main__":
    # train_and_save_model()
    model = load_model()
    untargeted_attack(model)
