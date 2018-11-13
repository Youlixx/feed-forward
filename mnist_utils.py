"""
Contains the main function to load a MNIST file as an array list
"""


import struct
import numpy


def read_idx(filename):
    """Loads the content of the idx (MNIST) file into a list of arrays
    :param filename: the path to the file
    :return: the list of loaded data"""

    file = open(filename, "rb")

    dims = struct.unpack('>HBB', file.read(4))[2]
    shape = tuple(struct.unpack('>I', file.read(4))[0] for _ in range(dims))

    return numpy.fromstring(file.read(), dtype=numpy.uint8).reshape(shape)
