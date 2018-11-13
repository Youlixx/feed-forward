"""
Contains the basic neural network functions such as save and load neural network
Also contains the main class of the neural network with the default stochastic gradient descent
algorithm implemented. The class also include some functions such as a random weights initiator
and a split function to generate sub networks (useful for auto encoders)
"""


import random
import numpy


class FeedForwardNetwork:
    """Class which instantiates a neural network. It contains all the basic functions to do
    forward and backward passes"""

    def __init__(self, layers):
        """The constructor of the class that generates an empty feed forward network
        :param layers: a list with the size (number of neurones) of each layers"""

        self.layers = layers

        self.weights = []
        self.biases = []

    def randomize(self, low, high):
        """Randomizes the weights and the biases of the network
        :param low: the lowest value possible for a weight / bias
        :param high: the highest value possible for a weight / bias"""

        for k in range(len(self.layers) - 1):
            weight = numpy.zeros((self.layers[k+1], self.layers[k]), dtype=float)
            bias = numpy.zeros((self.layers[k+1], 1), dtype=float)

            for i in range(self.layers[k+1]):
                for j in range(self.layers[k]):
                    weight[i, j] = random.uniform(low, high)

            for i in range(self.layers[k+1]):
                bias[i, 0] = random.uniform(low, high)

            self.weights.append(weight)
            self.biases.append(bias)

    def set_weights(self, weights, biases):
        """Changes the weights and biases of the network manually
        :param weights: a list of arrays (2D array)
        :param biases: a list of vectors (1D array)"""

        test = len(weights) == len(biases)

        for k in range(len(weights)):
            if not (weights[k].shape[0] == self.layers[k+1] and weights[k].shape[1] == self.layers[k]):
                test = False

            if not (biases[k].shape[0] == self.layers[k+1] and biases[k].shape[1] == 1):
                test = False

        assert test, "The sizes of the arrays do not match the specified layers sizes"

        self.weights = weights
        self.biases = biases

    def feed_forward(self, input_value):
        """Compute the output of the network for this input
        :param input_value: an array (1D vector) of the input data
        :return: an array (1D vector) of the output after processing the input through the layers"""

        output_value = input_value

        for k in range(len(self.layers) - 1):
            output_value = sigmoid(numpy.dot(self.weights[k], output_value) + self.biases[k])

        return output_value

    def back_propagation(self, input_value, expected_output):
        """Back propagation algorithm to compute the gradient of each weights and biases using the chain rule
        :param input_value: an array (1D vector) of the input data
        :param expected_output: an array (1D vector) of the expected output used to compute the error
        :return: the gradient of each weights and biases"""

        gradient_weights = []
        gradient_biases = []

        layer_state = input_value
        network_state = [input_value]

        network_sums = []

        for k in range(len(self.layers) - 1):
            weight = self.weights[k]
            bias = self.biases[k]

            gradient_weights.append(numpy.zeros(weight.shape))
            gradient_biases.append(numpy.zeros(bias.shape))

            layer_sum = numpy.dot(weight, layer_state) + bias
            network_sums.append(layer_sum)

            layer_state = sigmoid(layer_sum)
            network_state.append(layer_state)

        delta = cost_derivative(network_state[-1], expected_output) * sigmoid_derivative(network_sums[-1])

        gradient_weights[-1] = numpy.dot(delta, network_state[-2].transpose())
        gradient_biases[-1] = delta

        for k in range(2, len(self.layers)):
            layer_sum = network_sums[-k]
            derivative = sigmoid_derivative(layer_sum)
            delta = numpy.dot(self.weights[-k+1].transpose(), delta) * derivative

            gradient_weights[-k] = numpy.dot(delta, network_state[-k-1].transpose())
            gradient_biases[-k] = delta

        return gradient_weights, gradient_biases

    def update_batch(self, batch, learning_rate):
        """Applies the back propagation algorithm and then train the weights and biases over a simple batch
        :param batch: a list of couple of arrays (1D vector) with inputs / expected outputs
        :param learning_rate: a float (hyper parameter); the higher, the faster the learning is, but it can diverge
        :return: the average cost of the function before training"""

        gradient_weights = []
        gradient_biases = []

        cost = 0

        for k in range(len(self.layers) - 1):
            gradient_weights.append(numpy.zeros(self.weights[k].shape))
            gradient_biases.append(numpy.zeros(self.biases[k].shape))

        for sample in batch:
            back_prop = self.back_propagation(sample[0], sample[1])

            delta_gradient_weights = back_prop[0]
            delta_gradient_biases = back_prop[1]

            for i in range(len(gradient_weights)):
                gradient_weights[i] += delta_gradient_weights[i]
                gradient_biases[i] += delta_gradient_biases[i]

            cost += cost_function(self.feed_forward(sample[0]), sample[1])

        for k in range(len(self.layers) - 1):
            self.weights[k] = self.weights[k] - (learning_rate / len(batch)) * gradient_weights[k]
            self.biases[k] = self.biases[k] - (learning_rate / len(batch)) * gradient_biases[k]

        return cost

    def stochastic_gradient_descent(self, training_data, epochs, batch_size, learning_rate):
        """Applies the stochastic gradient descent on the network with the specified training data
        :param training_data: a list of couple of arrays (1D vector) with inputs / expected outputs
        :param epochs: the number of epochs (training batches)
        :param batch_size: the size of each batch used to train
        :param learning_rate: a float (hyper parameter); the higher, the faster the learning is, but it can diverge"""

        training_data = list(training_data)

        for k in range(epochs):
            random.shuffle(training_data)

            batches = []
            cost = 0

            for i in range(0, len(training_data), batch_size):
                batches.append(training_data[i:i + batch_size])

            for batch in batches:
                cost += self.update_batch(batch, learning_rate)

            print("Epoch {}/{} complete; average cost of the network over this epoch : {}".format(k+1, epochs, cost))

    def sub_network(self, start, end):
        """Creates a sub network
        :param start: the first layer of the sub network
        :param end: the last layer of the sub network
        :return: a new FeedForwardNetwork as a sub network"""

        sub_network = FeedForwardNetwork(self.layers[start:end])
        sub_network.set_weights(self.weights[start:end-1], self.biases[start:end-1])

        return sub_network


def cost_function(real_output, expected_output):
    """Computes the cost function for the real and the expected output
    :param real_output: an array (1D vector) of the output of the network
    :param expected_output: an array (1D vector) of the expected output for a given input
    :return: the cost value"""

    cost = 0

    for k in range(real_output.shape[0]):
        cost += (real_output[k] - expected_output[k]) ** 2

    return cost / (2 * real_output.shape[0])


def cost_derivative(real_output, expected_output):
    """Computes the derivative of the cost function 0.5 * (y - ÿ)²
    :param real_output: an array (1D vector) of the output of the network
    :param expected_output: an array (1D vector) of the expected output for a given input
    :return: the derivative of the cost function"""

    return real_output - expected_output


def sigmoid(x):
    """Computes the sigmoid function (activation function)
    :param x: a float
    :return: the sigmoid of x"""

    return 1.0 / (1.0 + numpy.exp(-x))


def sigmoid_derivative(x):
    """Computes the derivative of the sigmoid function (gradient descent)
    :param x: a float
    :return: the derivative of the sigmoid evaluated at x"""

    return sigmoid(x) * (1 - sigmoid(x))


def save_network(network, path, separators=";,"):
    """Saves a FeedForwardNetwork to a file
    :param network: the network to save
    :param path: the path to the file
    :param separators: the list of separators used in the file"""

    layers = network.layers
    weights = network.weights
    biases = network.biases

    file = open(path, "w")

    for k in range(len(layers)):
        if k != 0:
            file.write(separators[0])

        file.write(str(layers[k]))

    file.write("\n")

    for k in range(len(weights)):
        for i in range(weights[k].shape[0]):
            if i != 0:
                file.write(separators[0])

            for j in range(weights[k].shape[1]):
                if j != 0:
                    file.write(separators[1])

                file.write(str(weights[k][i, j]))

        file.write("\n")

    for k in range(len(biases)):
        for i in range(biases[k].shape[0]):
            if i != 0:
                file.write(separators[0])

            file.write(str(biases[k][i, 0]))

        file.write("\n")

    file.flush()
    file.close()


def load_network(path, separator=";,"):
    """Loads a FeedForwardNetwork from a file
    :param path: the path to file which contains the network
    :param separator: the list of separators used in the file
    :return: an instance of the loaded FeedForwardNetwork"""

    layers = []
    weights = []
    biases = []

    file = open(path, "r")

    layers_line = file.readline()

    for size in layers_line.split(separator[0]):
        layers.append(int(size))

    for k in range(len(layers) - 1):
        weight = numpy.zeros((layers[k+1], layers[k]), dtype=float)
        line = file.readline()
        nodes = line.split(separator[0])

        for i in range(len(nodes)):
            node = nodes[i].split(separator[1])

            for j in range(len(node)):
                weight[i, j] = float(node[j])

        weights.append(weight)

    for k in range(len(layers) - 1):
        bias = numpy.zeros((layers[k+1], 1), dtype=float)
        line = file.readline()
        nodes = line.split(separator[0])

        for i in range(len(nodes)):
            bias[i, 0] = float(nodes[i])

        biases.append(bias)

    network = FeedForwardNetwork(layers)
    network.set_weights(weights, biases)

    return network
