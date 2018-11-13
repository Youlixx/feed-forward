# Feed Forward Neural Network

Basic implementation of feed forward neural network using python.

## Getting started

These instructions will show you how to run a feed forward neural network and do testing on your local machine.

### Prerequisites

The weights of the neural network are stored in numpy arrays. You will also need matplotlib if you want to show the generated data of an auto encoder

## Running and test

The class FeedForwardNetwork contains all the stuff needed to generate the network and train it. It implements a basic stochastic gradient descent with a fixed learning rate.
Your file should start with the following lines

```
from feed_forward_network import FeedForwardNetwork   # loads the main class
from feed_forward_network import save_network         # loads the save function (optional)
from feed_forward_network import load_network         # loads the load function (optional)
from mnist_utils import read_idx                      # contains the function use to load data from MNIST set
```

### Create and train a new neural network

To create a new neural network, you have to specify the size of the layers (the neurons per layers). Initially, all the weights and biases of the network will be set with a value of 0.0.

```
layers = [784, 16, 16, 10]             # the sizes of the layers of the network
network = FeedForwardNetwork(layers)   # the network object
network.randomize(-1.0, 1.0)           # randomizes the weights/biases of the network with values between -1.0 and 1.0
```

To train the network, you have to specify the training data : a list which contains for each sample a tuple of the input and the expected output. You also have to specify the number of epochs, the number of batches per epochs and the learning rate.
Do not forget to save the state of the neural network if you want to use it later without training it again.

```
network.stochastic_gradient_descent(training_data, epoch, batches, learning_rate)   # trains the network
save_network(network, file_path)                                                    # saves the network to the specified file
```

### Load a neural network and test it

You can load a neural network to test it or to continue it's training.

```
network = load_network(file_path)   # loads the network
```

To get the output of neural network for a given input vector, simply do a forward pass.
Note that the network only accept column vectors and returns column vectors.

```
input_vector = numpy.array([1.0, 0.0, 0.0, 0.0]).transpose()   # creates an input vector
answer = network.feed_forward(input_vector)                    # returns the guess of the network for this input
```

### Trained network

You can use already trained networks stored in the trained_networks folder. All the trained network were trained on the MNIST data of Yann Lecun. You can download the data set on his [website](http://yann.lecun.com/exdb/mnist/)

```
network_predictor_1000e_100b_1.0lr.ffann -- trained to recognise the digit (1000 epochs of 100 batches with a learning rate of 1.0)
network_auto_encoder_100e_100b_1.0lr -- trained to generate the digit 5 (100 epochs of 100 batches with a learning rate of 1.0)
network_auto_encoder_100e0_100b_1.0lr -- trained to generate the digit 5 (1000 epochs of 100 batches with a learning rate of 1.0)
```



