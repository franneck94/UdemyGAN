import os

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam

from mnistCnn import build_cnn
from mnistData import MNIST
from plotting import plot_attack


np.random.seed(42)
PATH = os.path.abspath("C:/Users/Jan/OneDrive/_Coding/UdemyGAN")
MODELS_PATH = os.path.join(PATH, "models")
CNN_MODEL_PATH = os.path.join(MODELS_PATH, "mnist_cnn.h5")

mnist_data = MNIST()
x_train, y_train = mnist_data.get_train_set()
x_test, y_test = mnist_data.get_test_set()


def adversarial_noise(
    model: Model, image: np.ndarray, label: np.ndarray
) -> tf.Tensor:
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image, training=False)
        prediction = tf.reshape(prediction, (10,))
        loss = loss_object(label, prediction)
    # Get the gradients of the loss w.r.t. the input image
    gradient = tape.gradient(loss, image)
    # Get the sign of the gradients to create the noise
    signed_gradient = tf.sign(gradient)
    return signed_gradient


def train_and_save_model() -> None:
    model = build_cnn()
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"],
    )
    model.fit(
        x=x_train,
        y=y_train,
        verbose=1,
        batch_size=256,
        epochs=3,
        validation_data=(x_test, y_test),
    )
    model.save_weights(filepath=CNN_MODEL_PATH)


def load_model() -> Model:
    model = build_cnn()
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"],
    )
    model.load_weights(filepath=CNN_MODEL_PATH)
    return model


def targeted_attack(model: Model) -> None:
    sample_idx = np.random.randint(low=0, high=x_test.shape[0])
    image = np.array([x_test[sample_idx]])
    true_label = y_test[sample_idx]
    true_label_idx = np.argmax(true_label)
    y_pred = model.predict(image)[0]
    print("----Before Attack----")
    print(f"True class: {true_label_idx}")
    print(f"True class prob: {y_pred[true_label_idx]}")

    eps = 0.002
    image_adv = tf.convert_to_tensor(image, dtype=tf.float32)
    noise = tf.convert_to_tensor(np.zeros_like(image), dtype=tf.float32)
    target_label_idx = 9
    target_label = tf.one_hot(target_label_idx, 10)

    while np.argmax(y_pred) != target_label_idx:
        noise = adversarial_noise(model, image_adv, target_label)
        # if np.sum(noise) == 0.0:
        #     break
        image_adv = image_adv - eps * noise
        image_adv = tf.clip_by_value(image_adv, 0.0, 1.0)
        y_pred = model.predict(image_adv)[0]
        print(f"True class prob: {y_pred[true_label_idx]}")
        print(f"Targeted class prob: {y_pred[target_label_idx]}")

    plot_attack(image, image_adv.numpy())


if __name__ == "__main__":
    # train_and_save_model()
    model = load_model()
    targeted_attack(model)
