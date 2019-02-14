import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from HW_3 import utils


class WineClassifier:
    def __init__(self, input_dim, hidden_layer_dim, output_dim, seed) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.output_dim = output_dim

        np.random.seed(seed)
        tf.random.set_random_seed(seed)

        self.weights = {
            'hidden_layer': tf.Variable(tf.random_normal([self.input_dim, self.hidden_layer_dim])),
            'output_layer': tf.Variable(tf.random_normal([self.hidden_layer_dim, self.output_dim]))
        }

        self.bias = {
            'hidden_layer': tf.Variable(tf.random_normal([self.hidden_layer_dim])),
            'output_layer': tf.Variable(tf.random_normal([self.output_dim]))
        }

    def __get_hidden_layer(self, x):
        hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['hidden_layer']), self.bias['hidden_layer']))
        return hidden_layer

    def __get_output_layer(self, x):
        output_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['output_layer']), self.bias['output_layer']))
        return output_layer

    def __one_hot_encode(self, label):
        label = label.copy()
        label = label - 1
        unique_values = np.unique(label).shape[0]
        label_size = label.shape[0]
        hot_encoded_vector = np.zeros((label_size, unique_values), dtype=int)
        hot_encoded_vector[np.arange(label_size), label] = 1
        return hot_encoded_vector

    def predict(self, training_data, testing_data, learning_rate, epochs,
                display_step):

        training_data_features = training_data['features']
        testing_data_features = testing_data['features']
        training_data_labels = self.__one_hot_encode(training_data['labels'])

        X = tf.placeholder("float", [None, self.input_dim])
        Y = tf.placeholder("float", [None, self.output_dim])

        hidden_layer_output = self.__get_hidden_layer(X)
        output_layer_output = self.__get_output_layer(hidden_layer_output)

        y_pred = output_layer_output
        y_true = Y

        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for i in range(0, epochs):
                _, l = sess.run([optimizer, loss], feed_dict={
                    X: training_data_features,
                    Y: training_data_labels
                })

                if i % display_step == 0 or i == 1:
                    print('Step %i: Minibatch Loss: %f' % (i, l))

            training_prediction = sess.run(output_layer_output, feed_dict={X: training_data_features})
            training_prediction = sess.run(tf.add(tf.argmax(training_prediction, axis=1), 1))

            testing_prediction = sess.run(output_layer_output, feed_dict={X: testing_data_features})
            testing_prediction = sess.run(tf.add(tf.argmax(testing_prediction, axis=1), 1))

        return {
            "training": training_prediction,
            "testing": testing_prediction
        }


def demo_wine_classifier():
    data = utils.read_wine_data()
    classifier = WineClassifier(data['training']['features'].iloc[0].shape[0], 8, 3, seed=23)
    predicted = classifier.predict(data['training'], data['testing'], 0.01, 4000, 100)

    print("Training Accuracy", accuracy_score(data['training']['labels'], predicted['training']))
    print("Testing Accuracy", accuracy_score(data['testing']['labels'], predicted['testing']))


if __name__ == '__main__':
    demo_wine_classifier()
