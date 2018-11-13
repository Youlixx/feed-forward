"""
Creates a simple feed forward neural network. You can either choose to train a new neural network or
use an existing one to generate new samples.
"""


from feed_forward_network import FeedForwardNetwork
from feed_forward_network import save_network
from feed_forward_network import load_network

from mnist_utils import read_idx

import numpy


def create_training_data(image_array, label_array):
    """Generates the training data by linking the images to the labels
    :param image_array: the list of arrays (2D array)
    :param label_array: the list of labels
    :return: the training data to use for the stochastic gradient descent"""

    return [[array_to_vector(image_array[k]), vector_1d(label_array[k], 10)] for k in range(len(image_array))]


def array_to_vector(image):
    """Converts a 2D array into a vector (1D array)
    :param image: an array
    :return: the vector representation of the array"""

    vector = numpy.zeros((image.shape[0] * image.shape[1], 1))

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            vector[image.shape[1] * x + y] = image[x, y] / 255

    return vector


def vector_1d(index, size):
    """Generates the base vector Ei in a finite sized vector space
    :param index: an integer
    :param size: the size of the vector space
    :return: the base vector"""

    vector = numpy.zeros((size, 1))

    vector[index] = 1

    return vector


training = False

if training:
    print("Loading training data")
    training_images = read_idx("../hand_written_digits/train-images.idx3-ubyte")
    training_labels = read_idx("../hand_written_digits/train-labels.idx1-ubyte")

    print("Assigning labels to images")
    training_data = create_training_data(training_images, training_labels)

    print("Done! Creating neural network")
    layers = [784, 16, 16, 10]
    network = FeedForwardNetwork(layers)
    network.randomize(-1.0, 1.0)

    print("Start training over {} samples".format(len(training_data)))
    network.stochastic_gradient_descent(training_data, 1000, 100, 1.0)

    print("Done! Saving network")
    save_network(network, "trained_networks/network_predictor_1000e_100b_1.0lr.ffann")

else:
    print("Loading network")
    network = load_network("trained_networks/network_predictor_1000e_100b_1.0lr.ffann")

    print("Loading testing data")
    testing_images = read_idx("../hand_written_digits/t10k-images.idx3-ubyte")
    testing_labels = read_idx("../hand_written_digits/t10k-labels.idx1-ubyte")

    print("Done! Start testing")
    success = 0

    for i in range(len(testing_images)):
        answer = network.feed_forward(array_to_vector(testing_images[i]))

        guess = 0
        max_component = 0

        for j in range(answer.shape[0]):
            if answer[j, 0] > max_component:
                max_component = answer[j, 0]
                guess = j

        if guess == testing_labels[i]:
            success += 1

        print("Success rate of {}% over {} tests!".format(str(100.0 * success / (i + 1))[:5], i + 1))
