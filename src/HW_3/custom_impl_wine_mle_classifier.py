import numpy as np
from sklearn.metrics import accuracy_score

from HW_3 import utils
from HW_3.neural_network import ActivationFunction, NeuralNetwork, Loss


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

    nn.train(training_features, one_hot_training_labels, 0.01, 80, 10, 0.001)

    training_predicted = nn.predict(training_features)
    print("Training Accuracy", accuracy_score(training_labels, training_predicted))

    testing_predicted = nn.predict(testing_features)
    print("Testing Accuracy", accuracy_score(testing_labels, testing_predicted))


if __name__ == '__main__':
    demo_wine_mle_classifier()
