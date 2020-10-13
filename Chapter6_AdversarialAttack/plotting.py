import matplotlib.pyplot as plt


def plot_img(image, cmap="gray"):
    """Plot the image."""
    plt.imshow(image.squeeze(), cmap=cmap)
    plt.axis("off")
    plt.show()


def plot_attack(image_true, image_adversarial):
    """Plot the attack."""
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image_true.reshape((28, 28)), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("image_adversarial")
    plt.imshow(image_adversarial.reshape((28, 28)), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Difference")
    difference = image_adversarial - image_true
    plt.imshow(difference.reshape((28, 28)), cmap="gray")
    plt.axis("off")
    plt.show()
