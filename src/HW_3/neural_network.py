from enum import Enum

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from tensorflow import sigmoid

from HW_3 import utils


class ActivationFunction(Enum):
    RELU = 1
    SIGMOID = 2
    SOFT_MAX = 3


class Loss(Enum):
    MEAN_SQUARE_ERROR = 1
    MAXIMUM_LIKELIHOOD_CROSS_ENTROPY = 2


class NeuralNetwork:
    def __init__(self, input_dim, hidden_layer_dim, output_dim, activation_functions, loss, seed) -> None:
        super().__init__()
        self.loss = loss
        self.input_dim = input_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.output_dim = output_dim
        self.activation_functions = activation_functions

        np.random.seed(seed)
        self.__init_weights()
        self.__init_bias()

    def __init_weights(self):
        self.weights = {
            'hidden_layer': np.random.random((self.input_dim, self.hidden_layer_dim)),
            'output_layer': np.random.random((self.hidden_layer_dim, self.output_dim))
        }

    def __init_bias(self):
        self.bias = {
            'hidden_layer': np.random.random(self.hidden_layer_dim),
            'output_layer': np.random.random(self.output_dim)
        }

    def __relu(self, x):
        x = x.copy()
        x[x < 0] = 0
        return x

    def __get_activation_function(self, layer):
        activation_function = self.activation_functions[layer]
        if activation_function == ActivationFunction.RELU:
            return self.__relu
        elif activation_function == ActivationFunction.SIGMOID:
            return lambda x: sigmoid(x)
        elif activation_function == ActivationFunction.SOFT_MAX:
            return lambda x: softmax(x)
        else:
            raise Exception("Unknown Activation Function")

    def predict(self, features):
        hidden_layer_x_w = np.matmul(features, self.weights['hidden_layer'])
        hidden_layer_output = self.__get_activation_function('hidden_layer')(hidden_layer_x_w)

        output_layer_x_w = np.matmul(hidden_layer_output, self.weights['output_layer'])
        output_layer_output = self.__get_activation_function('output_layer')(output_layer_x_w)

        return output_layer_output

    def train(self, training_data):
        training_features = training_data['features']
        training_labels = training_data['labels']


def demo_wine_classifier():
    data = utils.read_wine_data()
    training_data = data['training']

    input_dim = training_data[0].shape[0]
    hidden_layer_dim = 8
    output_layer_dim = 3

    activation_functions = {
        'hidden_layer': ActivationFunction.SIGMOID,
        'output_layer': ActivationFunction.SIGMOID
    }

    nn = NeuralNetwork(input_dim, hidden_layer_dim, output_layer_dim, activation_functions, Loss.MEAN_SQUARE_ERROR, 23)
    training_predicted = nn.predict(training_data)
    print("Training Accuracy", accuracy_score(training_data['labels'], training_predicted))

    testing_data = data['testing']
    testing_predicted = nn.predict(testing_data)
    print("Testing Accuracy", accuracy_score(testing_data['labels'], testing_predicted))


if __name__ == '__main__':
    demo_wine_classifier()
