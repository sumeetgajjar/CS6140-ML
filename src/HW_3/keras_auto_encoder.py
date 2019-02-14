import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model

ORIGINAL_DIM = 8
ENCODING_DIM = 3


def get_input_for_encoder():
    features = np.repeat(np.zeros((1, ORIGINAL_DIM), dtype=np.int), ORIGINAL_DIM, axis=0)
    for i in range(0, 8):
        features[i][i] = 1
    return features


np.random.seed(11)

training_features = get_input_for_encoder()

encoder_input = Input(shape=training_features.shape)
encoded = Dense(ENCODING_DIM, activation='sigmoid')(encoder_input)
encoder = Model(encoder_input, encoded)

decoder_input = Input(shape=(ENCODING_DIM,))
decoded = Dense(ORIGINAL_DIM, activation='sigmoid')(decoder_input)
decoder = Model(decoder_input, decoded)

network_input = Input(shape=training_features[0].shape)
encoder_output = encoder(network_input)
decoder_output = decoder(encoder_output)

model = Model(network_input, decoder_output)
model.compile(optimizer='sgd', loss='mse')
tensor_board_callback = TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=0, write_grads=True, write_images=True)

model.fit(training_features, training_features, batch_size=8, epochs=100, callbacks=[tensor_board_callback])

predict = model.predict(training_features)
print(predict)
