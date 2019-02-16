from HW_3 import utils
from HW_3.neural_network import ActivationFunction, NeuralNetwork, Loss


def demo_auto_encoder():
    training_features = utils.get_input_for_encoder(8)
    training_labels = training_features.copy()

    input_dim = training_features[0].shape[0]
    hidden_layer_dim = 3
    output_layer_dim = 8

    activation_functions = {
        'hidden_layer': ActivationFunction.SIGMOID,
        'output_layer': ActivationFunction.SIGMOID
    }

    nn = NeuralNetwork(input_dim, hidden_layer_dim, output_layer_dim, activation_functions, Loss.MEAN_SQUARE_ERROR,
                       123)

    nn.train(training_features, training_labels, 0.2, 2000, 100, 0.001)

    predicted = nn.predict(training_features)
    print("Predicted ", predicted)


if __name__ == '__main__':
    demo_auto_encoder()
