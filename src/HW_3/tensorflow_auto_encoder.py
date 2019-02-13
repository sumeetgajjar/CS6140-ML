import numpy as np
import tensorflow as tf

from HW_3 import utils

ORIGINAL_DIM = 8
ENCODING_DIM = 3

np.random.seed(11)
tf.random.set_random_seed(11)

training_features = utils.get_input_for_encoder(ORIGINAL_DIM)

learning_rate = 0.01
num_steps = 4000
batch_size = 8

display_step = 100

num_hidden_1 = 3
num_input = 8

X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_1, num_input]))
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b1': tf.Variable(tf.random_normal([num_input]))
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    return layer_1


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    return layer_1


encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1, num_steps + 1):
        _, l = sess.run([optimizer, loss], feed_dict={X: training_features})
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    predict = sess.run(decoder_op, feed_dict={X: training_features})
    print(predict)
    print(np.round(predict))
