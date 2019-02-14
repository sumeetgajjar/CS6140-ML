import numpy as np
import tensorflow as tf

from HW_3 import utils


class AutoEncoder:

    def __init__(self, input_dim, encoded_dim, seed) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.encoded_dim = encoded_dim
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_random_seed(seed)

        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.input_dim, self.encoded_dim])),
            'decoder_h1': tf.Variable(tf.random_normal([self.encoded_dim, self.input_dim]))
        }

        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.encoded_dim])),
            'decoder_b1': tf.Variable(tf.random_normal([self.input_dim]))
        }

    def __get_encoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']), self.biases['encoder_b1']))
        return layer_1

    def __get_decoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']), self.biases['decoder_b1']))
        return layer_1

    def predict(self, training_features, display_step, learning_rate, epochs):

        X = tf.placeholder("float", [None, self.input_dim])
        encoder_output = self.__get_encoder(X)
        decoder_output = self.__get_decoder(encoder_output)

        y_pred = decoder_output
        y_true = X

        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for i in range(0, epochs):
                _, l = sess.run([optimizer, loss], feed_dict={X: training_features})
                if i % display_step == 0 or i == 1:
                    print('Step %i: Minibatch Loss: %f' % (i, l))

            predict = sess.run(decoder_output, feed_dict={X: training_features})
            predict = sess.run(tf.argmax(predict, axis=1))
            true = sess.run(tf.argmax(training_features, axis=1))

        return {
            "Truth": true,
            "Predicted": predict
        }


def demo_auto_encoder():
    training_features = utils.get_input_for_encoder(8)
    auto_encoder = AutoEncoder(input_dim=8, encoded_dim=3, seed=11)
    predicted = auto_encoder.predict(training_features, display_step=100, learning_rate=0.01, epochs=4000)
    print(predicted)


if __name__ == '__main__':
    demo_auto_encoder()
