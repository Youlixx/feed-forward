"""
Contains the class use to perform a PCA (principal component analysis). Should be used to make the
data uncorrelated (typical use : generating new sample from the decoder part of an auto encoder)
"""


import numpy


class PrincipalComponentAnalysis:
    """Class which instantiates a PCA (principal component analysis). It allows to transform a vector of
    the latent space into another vector of the same dimension but with less correlation"""

    def __init__(self, encoder, sampled_data):
        """Applies the PCA algorithm over the random sampled data to reduce the correlation of the output
        :param encoder: the network used to encode the input
        :param sampled_data: list of random inputs (1D arrays)"""

        data = numpy.zeros((len(sampled_data), encoder.layers[-1]))

        for i in range(data.shape[0]):
            output = encoder.feed_forward(sampled_data[i])

            for j in range(data.shape[1]):
                data[i, j] = output[j]

        mean = [0.0] * data.shape[1]

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                mean[j] += data[i, j]

        for j in range(data.shape[1]):
            mean[j] /= data.shape[0]

        adjusted = numpy.zeros(data.shape)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                adjusted[i, j] = data[i, j] - mean[j]

        covariance = numpy.zeros((data.shape[1], data.shape[1]))

        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                variance = 0

                for k in range(data.shape[0]):
                    variance += (data[k, i] - mean[i]) * (data[k, j] - mean[j])

                covariance[i, j] = variance / (data.shape[0] - 1)

        self.transform_data = numpy.linalg.eig(covariance)

    def transform_vector(self, vector, standard_deviation=2):
        """Transforms the vector from the canonical basis to the les correlated basis
        :param vector: the vector to transform (1D array)
        :param standard_deviation: the standard deviation used to recenter the vector
        :return: the new vector after transform"""

        eig_values = self.transform_data[0]
        centered = numpy.zeros((len(eig_values), 1))

        for i in range(centered.shape[0]):
            centered[i, 0] = eig_values[i] + (vector[i, 0] - 0.5) * 2 * standard_deviation

        return numpy.dot(numpy.transpose(self.transform_data[1]), centered)
