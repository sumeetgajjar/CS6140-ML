import tensorflow as tf
from sklearn.metrics import accuracy_score

from HW_3 import utils
from HW_3.tensor_flow_wine_classifier import WineClassifier


class MleWineClassifier(WineClassifier):

    def __init__(self, input_dim, hidden_layer_dim, output_dim, seed) -> None:
        super().__init__(input_dim, hidden_layer_dim, output_dim, seed)

    def get_hidden_layer(self, x):
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, self.weights['hidden_layer']), self.bias['hidden_layer']))
        return hidden_layer

    def get_output_layer(self, x):
        output_layer = tf.add(tf.matmul(x, self.weights['output_layer']), self.bias['output_layer'])
        return output_layer

    def get_loss(self, y_pred, y_true):
        return tf.losses.softmax_cross_entropy(y_true, y_pred)


def demo_maximum_likelihood_wine_classifier():
    print("Maximum Likelihood Wine Classifier")
    data = utils.read_wine_data()
    classifier = MleWineClassifier(data['training']['features'].iloc[0].shape[0], 4, 3, seed=11)
    predicted = classifier.predict(data['training'], data['testing'], 0.01, 1000, 100)

    print("Training Accuracy", accuracy_score(data['training']['labels'], predicted['training']))
    print("Testing Accuracy", accuracy_score(data['testing']['labels'], predicted['testing']))


if __name__ == '__main__':
    demo_maximum_likelihood_wine_classifier()
