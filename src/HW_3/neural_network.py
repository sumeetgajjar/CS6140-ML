from enum import Enum

import numpy as np
from scipy.special import softmax, expit
from sklearn.metrics import accuracy_score

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

    def relu(self, x):
        x = x.copy()
        x[x <= 0] = 0
        return x

    def get_activation_function(self, layer):
        activation_function = self.activation_functions[layer]
        if activation_function == ActivationFunction.RELU:
            return self.relu
        elif activation_function == ActivationFunction.SIGMOID:
            return lambda x: expit(x)
        elif activation_function == ActivationFunction.SOFT_MAX:
            return lambda x: softmax(x, axis=1)
        else:
            raise Exception("Unknown Activation Function")

    def get_hidden_layer_output(self, input_features):
        hidden_layer_x_w = np.matmul(input_features, self.weights['hidden_layer'])
        hidden_layer_output = self.get_activation_function('hidden_layer')(hidden_layer_x_w)
        return hidden_layer_output

    def get_output_layer_output(self, hidden_layer_output):
        output_layer_x_w = np.matmul(hidden_layer_output, self.weights['output_layer'])
        output_layer_output = self.get_activation_function('output_layer')(output_layer_x_w)
        return output_layer_output

    def forward_propagate(self, features):
        hidden_layer_output = self.get_hidden_layer_output(features)
        output_layer_output = self.get_output_layer_output(hidden_layer_output)
        return output_layer_output

    def predict(self, features):
        y_pred = self.forward_propagate(features)
        return np.argmax(y_pred, axis=1) + 1

    def get_square_loss(self, t_k, z_k):
        j_w = 0.5 * np.sum(np.square(t_k - z_k))
        return j_w

    def get_cross_entropy_loss(self, t_k, z_k):
        log_z_k = []
        log_one_minus_z_k = []
        for z in z_k:
            temp1 = []
            temp2 = []
            for label in z:
                if label == 0:
                    temp1.append(label)
                else:
                    temp1.append(np.log(label))

                if 1 - label == 0:
                    temp2.append(label)
                else:
                    temp2.append(np.log(1 - label))

            log_z_k.append(temp1)
            log_one_minus_z_k.append(temp2)

        log_z_k = np.array(log_z_k)
        log_one_minus_z_k = np.array(log_one_minus_z_k)

        return -np.sum((t_k * log_z_k) + ((1 - t_k) * log_one_minus_z_k))

    def get_loss(self, t_k, z_k):
        if self.loss == Loss.MEAN_SQUARE_ERROR:
            return self.get_square_loss(t_k, z_k)
        elif self.loss == Loss.MAXIMUM_LIKELIHOOD_CROSS_ENTROPY:
            return self.get_cross_entropy_loss(t_k, z_k)

    def relu_derivative(self, output):
        output = output.copy()
        output[output <= 0] = 0
        output[output > 0] = 1
        return output

    def get_derivative(self, output, layer):
        activation_function = self.activation_functions[layer]
        if activation_function == ActivationFunction.RELU:
            return self.relu_derivative(output)
        elif activation_function == ActivationFunction.SIGMOID:
            return output * (1 - output)
        elif activation_function == ActivationFunction.SOFT_MAX:
            return lambda x: softmax(x)
        else:
            raise Exception("Unknown Activation Function")

    def get_outer_layer_derivative(self, z):
        return self.get_derivative(z, 'output_layer')

    def get_hidden_layer_derivative(self, y):
        return self.get_derivative(y, 'hidden_layer')

    def update_weights_for_reduced_mean_loss(self, learning_rate, training_features, training_labels, index, y, z):
        f_prime_net_k = self.get_outer_layer_derivative(z)

        outer_w = self.weights['output_layer']
        for j in range(self.hidden_layer_dim):
            for k in range(self.output_dim):
                delta_outer = (
                        learning_rate * (training_labels[index][k] - z[index][k]) * f_prime_net_k[index][k] * y[index][
                    j])
                outer_w[j][k] = outer_w[j][k] + delta_outer

        f_prime_net_j = self.get_hidden_layer_derivative(y)

        hidden_w = self.weights['hidden_layer']
        for i in range(self.input_dim):
            for j in range(self.hidden_layer_dim):
                temp = 0
                for k in range(self.output_dim):
                    temp = temp + (training_labels[index][k] - z[index][k]) * f_prime_net_k[index][k] * outer_w[j][k]

                delta_hidden = (learning_rate * temp * f_prime_net_j[index][j] * training_features[index][i])
                hidden_w[i][j] = hidden_w[i][j] + delta_hidden

        return hidden_w, outer_w

    def update_weights_for_soft_max_cross_entropy_loss(self, learning_rate, training_features,
                                                       training_labels, index, y, z):

        outer_w = self.weights['output_layer']
        for j in range(self.hidden_layer_dim):
            for k in range(self.output_dim):
                outer_w[j][k] = outer_w[j][k] + (
                        learning_rate * y[index][j] * (training_labels[index][k] - z[index][k]))

        hidden_w = self.weights['hidden_layer']
        for i in range(self.input_dim):
            for j in range(self.hidden_layer_dim):
                temp = 0
                if training_features[index][j] >= 0:
                    for k in range(self.output_dim):
                        temp = temp + (training_features[index][i] * (training_labels[index][k] - z[index][k]) *
                                       outer_w[j][k])

                hidden_w[i][j] = hidden_w[i][j] + (learning_rate * temp)

        return hidden_w, outer_w

    def train(self, training_features, training_labels, learning_rate, epochs, display_step, epsilon):
        for iteration in range(1, epochs + 1):

            printed = False

            for x in range(training_features.shape[0]):

                y = self.get_hidden_layer_output(training_features)
                z = self.get_output_layer_output(y)
                j_w = self.get_loss(training_labels, z)
                if not printed and (iteration % display_step == 0 or iteration == 1):
                    printed = True
                    print('Step %i: Minibatch Loss: %f' % (iteration, j_w))

                if j_w < epsilon:
                    print("Loss is less than Epsilon: J={}, Epsilon={}".format(j_w, epsilon))
                    return

                if self.loss == Loss.MEAN_SQUARE_ERROR:
                    hidden_w, outer_w = self.update_weights_for_reduced_mean_loss(learning_rate, training_features,
                                                                                  training_labels, x, y, z)
                else:
                    hidden_w, outer_w = self.update_weights_for_soft_max_cross_entropy_loss(learning_rate,
                                                                                            training_features,
                                                                                            training_labels, x, y, z)

                self.weights['hidden_layer'] = hidden_w
                self.weights['output_layer'] = outer_w


def demo_wine_mle_classifier():
    data = utils.read_wine_data()

    training_features = data['training']['features']
    training_labels = data['training']['labels']
    one_hot_training_labels = utils.one_hot_encode_wine_classifier_labels(training_labels)

    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    combined_features = np.concatenate((training_features, testing_features))
    normalized_features = utils.normalize_data_using_zero_mean_unit_variance(combined_features)

    training_features = normalized_features[:training_features.shape[0]]
    testing_features = normalized_features[training_features.shape[0]:]

    training_features = utils.prepend_one_to_feature_vectors(training_features)
    testing_features = utils.prepend_one_to_feature_vectors(testing_features)

    input_dim = training_features[0].shape[0]
    hidden_layer_dim = 8
    output_layer_dim = 3

    activation_functions = {
        'hidden_layer': ActivationFunction.RELU,
        'output_layer': ActivationFunction.SOFT_MAX
    }

    nn = NeuralNetwork(input_dim, hidden_layer_dim, output_layer_dim, activation_functions,
                       Loss.MAXIMUM_LIKELIHOOD_CROSS_ENTROPY, 123)

    nn.train(training_features, one_hot_training_labels, 0.03, 40, 10, 0.001)

    training_predicted = nn.predict(training_features)
    print("Training Accuracy", accuracy_score(training_labels, training_predicted))

    testing_predicted = nn.predict(testing_features)
    print("Testing Accuracy", accuracy_score(testing_labels, testing_predicted))


if __name__ == '__main__':
    demo_wine_mle_classifier()
