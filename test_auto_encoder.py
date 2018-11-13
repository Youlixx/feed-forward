"""
Creates an auto encoder. You can either choose to train a new neural network or use an
existing one to generate new samples.
"""


from feed_forward_network import FeedForwardNetwork
from feed_forward_network import save_network
from feed_forward_network import load_network

from principal_component_analysis import PrincipalComponentAnalysis
from mnist_utils import read_idx

import random
import numpy
import matplotlib.pyplot as plot


def create_training_data(image_array, label_array, index):
    """Generates the training data by linking the images to the labels
    :param image_array: the list of arrays (2D array)
    :param label_array:  the list of labels
    :param index: the number of the set
    :return: the training data to use for the stochastic gradient descent"""

    data_set = []

    for i in range(len(image_array)):
        if label_array[i] == index:
            image = array_to_vector(image_array[i])

            data_set.append([image, image])

    return data_set


def array_to_vector(image):
    """Converts a 2D array into a vector (1D array)
    :param image: an array
    :return: the vector representation of the array"""

    vector = numpy.zeros((image.shape[0] * image.shape[1], 1))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            vector[image.shape[1] * i + j] = image[i, j] / 255

    return vector


def vector_to_array(vector, width, height):
    """Converts a 1D array into a 2D array
    :param vector: the vector
    :param width: the width of the array
    :param height: the height of the array
    :return: the image array"""

    image = numpy.zeros((width, height))

    for i in range(width):
        for j in range(height):
            image[i, j] = vector[height * i + j]

    return image


def random_vector(size):
    """Generates a random vector (1D array)
    :param size: the dimension of the vector
    :return: the random vector"""

    vector = numpy.zeros((size, 1))

    for i in range(size):
        vector[i, 0] = random.uniform(0, 1)

    return vector


training = False

if training:
    print("Loading training data")
    training_images = read_idx("../hand_written_digits/train-images.idx3-ubyte")
    training_labels = read_idx("../hand_written_digits/train-labels.idx1-ubyte")

    print("Assigning labels to images")
    training_data = create_training_data(training_images, training_labels, 5)

    print("Done! Creating neural network")
    layers = [784, 392, 196, 98, 5, 98, 196, 392, 784]
    network = FeedForwardNetwork(layers)
    network.randomize(-1.0, 1.0)

    print("Start training over {} samples".format(len(training_data)))
    network.stochastic_gradient_descent(training_data, 1000, 100, 1.0)

    print("Done! Saving network")
    save_network(network, "trained_network/network_auto_encoder_1000e_100b_1.0lr.ffann")

else:
    print("Loading network")
    network = load_network("trained_networks/network_auto_encoder_1000e_100b_1.0lr.ffann")

    encoder = network.sub_network(0, 5)
    decoder = network.sub_network(4, 9)

    sampled_data = []

    print("Loading training data")
    testing_images = read_idx("../hand_written_digits/t10k-images.idx3-ubyte")
    testing_labels = read_idx("../hand_written_digits/t10k-labels.idx1-ubyte")

    for k in range(len(testing_images)):
        if testing_labels[k] == 5:
            sampled_data.append(array_to_vector(testing_images[k]))

    print("Applying PCA over {} samples".format(len(sampled_data)))
    pca = PrincipalComponentAnalysis(encoder, sampled_data)

    print("Done!")
    figure = plot.figure()

    ims = []
    input_vector = numpy.array([[0.5, 0.5, 0.5, 0.5, 0.5]]).transpose()
    step = 0.01

    for k in range(60):
        decoded = vector_to_array(decoder.feed_forward(pca.transform_vector(numpy.array([[k / 60, 0.5, 0.5, 0.5, 0.5]]).transpose(), standard_deviation=2)), 28, 28)
        im = plot.imshow(numpy.asarray(decoded))
        plot.pause(0.03)
        plot.draw()
