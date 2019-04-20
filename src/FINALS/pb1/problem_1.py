import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense


def add_noise(x, noise_factor=0.2):
    x = x + np.random.randn(*x.shape) * noise_factor
    x = x.clip(0., 1.)
    return x


SEED = 11
np.random.seed(SEED)
tf.random.set_random_seed(SEED)

x_train = np.load('x_train.npy')
x_test_usetodebug = np.load('x_test_usetodebug.npy')
x_test_noisy = np.load('x_test_noisy.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=SEED)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test_noisy)
print("Testing Accuracy using Logistic Regression before cleaning:", accuracy_score(y_test, y_pred))

# start of auto encoder for denoising the data

inputs = Input(shape=(784,))  # 28*28 flatten
enc_fc = Dense(144, activation='sigmoid')  # to 32 data points
encoded = enc_fc(inputs)

dec_fc = Dense(784, activation='sigmoid')  # to 784 data points
decoded = dec_fc(encoded)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

x_train_1, x_valid_1 = train_test_split(x_train, test_size=500)

x_train_1_noisy = add_noise(x_train_1, noise_factor=0.44)
x_valid_1_noisy = add_noise(x_valid_1, noise_factor=0.44)

autoencoder.fit(x_train_1_noisy, x_train_1,  # data and label are the same
                epochs=100,
                batch_size=1000, validation_data=(x_valid_1_noisy, x_valid_1))

x_test_noisy_processed = autoencoder.predict(x_test_noisy)

# Plot figures
n = 10
for i in range(n):
    i = i + 1
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test_noisy[i - 1].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noisy
    ax = plt.subplot(2, n, i + n)
    plt.imshow(x_train_1_noisy[i - 1].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# end of auto encoder for denoising the data
y_test_pred = clf.predict(x_test_noisy_processed)

print("Testing Accuracy using Logistic Regression after cleaning:", accuracy_score(y_test, y_test_pred))
