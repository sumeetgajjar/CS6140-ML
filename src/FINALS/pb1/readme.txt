I used an auto encoder to denoise the testing data set.

For training the data autoencoder,
I added noise to the x_train data to generate x_train_noisy dataset.
Then I trained my autoencoder with x_train_noisy as input and x_train as output.
Hence my autoencoder has learned to take in noisy data and generate clean data.

Using this method I achieved accuracy of 88.29 percent using the given LogisticRegression classifier.


I used K nearest neighbors classifier to directly classify the noisy data.
The reason for using KNN is that it's a local classifier. And even though the testing data contains the noise, the
majority of the content in the features is not noise knn will select the correct labels.

Using this classifier I achieved an accuracy of 94.43 percent.

Output:

/home/sumeet/PycharmProjects/CS6140-ML/venv/bin/python /home/sumeet/PycharmProjects/CS6140-ML/src/FINALS/pb1/problem_1.py
/home/sumeet/PycharmProjects/CS6140-ML/venv/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:758: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
  "of iterations.", ConvergenceWarning)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Testing Accuracy using Logistic Regression before cleaning: 0.2999
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

WARNING:tensorflow:From /home/sumeet/PycharmProjects/CS6140-ML/venv/lib/python3.6/site-packages/tensorflow/python/ops/resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /home/sumeet/PycharmProjects/CS6140-ML/venv/lib/python3.6/site-packages/tensorflow/python/keras/utils/losses_utils.py:170: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Train on 59500 samples, validate on 500 samples
WARNING:tensorflow:From /home/sumeet/PycharmProjects/CS6140-ML/venv/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Epoch 1/100
2019-04-20 13:01:46.457773: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-04-20 13:01:46.594524: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-04-20 13:01:46.595039: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0xfe16e0 executing computations on platform CUDA. Devices:
2019-04-20 13:01:46.595059: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): GeForce GTX 1060, Compute Capability 6.1
2019-04-20 13:01:46.616936: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2208000000 Hz
2019-04-20 13:01:46.618233: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x1024dd0 executing computations on platform Host. Devices:
2019-04-20 13:01:46.618273: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2019-04-20 13:01:46.618523: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties:
name: GeForce GTX 1060 major: 6 minor: 1 memoryClockRate(GHz): 1.48
pciBusID: 0000:01:00.0
totalMemory: 5.94GiB freeMemory: 5.44GiB
2019-04-20 13:01:46.618537: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-04-20 13:01:46.619204: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-04-20 13:01:46.619215: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0
2019-04-20 13:01:46.619220: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N
2019-04-20 13:01:46.619386: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5270 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2019-04-20 13:01:47.219209: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcublas.so.10.0 locally
59500/59500 [==============================] - 2s 26us/sample - loss: 0.1106 - val_loss: 0.0874
Epoch 2/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0856 - val_loss: 0.0852
Epoch 3/100
59500/59500 [==============================] - 1s 9us/sample - loss: 0.0820 - val_loss: 0.0802
Epoch 4/100
59500/59500 [==============================] - 1s 9us/sample - loss: 0.0768 - val_loss: 0.0750
Epoch 5/100
59500/59500 [==============================] - 1s 9us/sample - loss: 0.0719 - val_loss: 0.0701
Epoch 6/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0672 - val_loss: 0.0657
Epoch 7/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0632 - val_loss: 0.0620
Epoch 8/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0599 - val_loss: 0.0590
Epoch 9/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0571 - val_loss: 0.0564
Epoch 10/100
59500/59500 [==============================] - 1s 11us/sample - loss: 0.0547 - val_loss: 0.0541
Epoch 11/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0526 - val_loss: 0.0521
Epoch 12/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0507 - val_loss: 0.0503
Epoch 13/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0489 - val_loss: 0.0486
Epoch 14/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0473 - val_loss: 0.0471
Epoch 15/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0457 - val_loss: 0.0456
Epoch 16/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0443 - val_loss: 0.0443
Epoch 17/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0430 - val_loss: 0.0430
Epoch 18/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0418 - val_loss: 0.0418
Epoch 19/100
59500/59500 [==============================] - 1s 9us/sample - loss: 0.0406 - val_loss: 0.0407
Epoch 20/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0396 - val_loss: 0.0398
Epoch 21/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0386 - val_loss: 0.0388
Epoch 22/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0377 - val_loss: 0.0380
Epoch 23/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0369 - val_loss: 0.0372
Epoch 24/100
59500/59500 [==============================] - 1s 11us/sample - loss: 0.0361 - val_loss: 0.0364
Epoch 25/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0353 - val_loss: 0.0357
Epoch 26/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0346 - val_loss: 0.0350
Epoch 27/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0340 - val_loss: 0.0344
Epoch 28/100
59500/59500 [==============================] - 1s 9us/sample - loss: 0.0333 - val_loss: 0.0338
Epoch 29/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0328 - val_loss: 0.0332
Epoch 30/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0322 - val_loss: 0.0327
Epoch 31/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0317 - val_loss: 0.0322
Epoch 32/100
59500/59500 [==============================] - 1s 11us/sample - loss: 0.0312 - val_loss: 0.0317
Epoch 33/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0307 - val_loss: 0.0313
Epoch 34/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0303 - val_loss: 0.0309
Epoch 35/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0299 - val_loss: 0.0305
Epoch 36/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0295 - val_loss: 0.0301
Epoch 37/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0291 - val_loss: 0.0298
Epoch 38/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0288 - val_loss: 0.0295
Epoch 39/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0284 - val_loss: 0.0291
Epoch 40/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0281 - val_loss: 0.0288
Epoch 41/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0278 - val_loss: 0.0285
Epoch 42/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0275 - val_loss: 0.0282
Epoch 43/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0272 - val_loss: 0.0280
Epoch 44/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0269 - val_loss: 0.0277
Epoch 45/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0267 - val_loss: 0.0275
Epoch 46/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0264 - val_loss: 0.0272
Epoch 47/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0262 - val_loss: 0.0270
Epoch 48/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0259 - val_loss: 0.0268
Epoch 49/100
59500/59500 [==============================] - 1s 11us/sample - loss: 0.0257 - val_loss: 0.0266
Epoch 50/100
59500/59500 [==============================] - 1s 11us/sample - loss: 0.0255 - val_loss: 0.0264
Epoch 51/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0253 - val_loss: 0.0262
Epoch 52/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0251 - val_loss: 0.0260
Epoch 53/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0249 - val_loss: 0.0258
Epoch 54/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0247 - val_loss: 0.0257
Epoch 55/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0245 - val_loss: 0.0255
Epoch 56/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0243 - val_loss: 0.0253
Epoch 57/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0242 - val_loss: 0.0252
Epoch 58/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0240 - val_loss: 0.0250
Epoch 59/100
59500/59500 [==============================] - 1s 11us/sample - loss: 0.0239 - val_loss: 0.0249
Epoch 60/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0237 - val_loss: 0.0247
Epoch 61/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0236 - val_loss: 0.0246
Epoch 62/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0234 - val_loss: 0.0244
Epoch 63/100
59500/59500 [==============================] - 1s 11us/sample - loss: 0.0233 - val_loss: 0.0243
Epoch 64/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0232 - val_loss: 0.0242
Epoch 65/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0230 - val_loss: 0.0241
Epoch 66/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0229 - val_loss: 0.0240
Epoch 67/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0228 - val_loss: 0.0238
Epoch 68/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0227 - val_loss: 0.0237
Epoch 69/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0226 - val_loss: 0.0236
Epoch 70/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0224 - val_loss: 0.0235
Epoch 71/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0223 - val_loss: 0.0234
Epoch 72/100
59500/59500 [==============================] - 1s 9us/sample - loss: 0.0222 - val_loss: 0.0233
Epoch 73/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0221 - val_loss: 0.0233
Epoch 74/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0220 - val_loss: 0.0231
Epoch 75/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0219 - val_loss: 0.0230
Epoch 76/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0218 - val_loss: 0.0230
Epoch 77/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0217 - val_loss: 0.0229
Epoch 78/100
59500/59500 [==============================] - 1s 9us/sample - loss: 0.0217 - val_loss: 0.0228
Epoch 79/100
59500/59500 [==============================] - 1s 9us/sample - loss: 0.0216 - val_loss: 0.0227
Epoch 80/100
59500/59500 [==============================] - 1s 11us/sample - loss: 0.0215 - val_loss: 0.0226
Epoch 81/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0214 - val_loss: 0.0226
Epoch 82/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0213 - val_loss: 0.0225
Epoch 83/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0212 - val_loss: 0.0224
Epoch 84/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0212 - val_loss: 0.0223
Epoch 85/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0211 - val_loss: 0.0223
Epoch 86/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0210 - val_loss: 0.0222
Epoch 87/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0209 - val_loss: 0.0221
Epoch 88/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0209 - val_loss: 0.0221
Epoch 89/100
59500/59500 [==============================] - 1s 11us/sample - loss: 0.0208 - val_loss: 0.0220
Epoch 90/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0207 - val_loss: 0.0220
Epoch 91/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0207 - val_loss: 0.0219
Epoch 92/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0206 - val_loss: 0.0218
Epoch 93/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0205 - val_loss: 0.0218
Epoch 94/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0205 - val_loss: 0.0217
Epoch 95/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0204 - val_loss: 0.0216
Epoch 96/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0204 - val_loss: 0.0216
Epoch 97/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0203 - val_loss: 0.0216
Epoch 98/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0202 - val_loss: 0.0215
Epoch 99/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0202 - val_loss: 0.0215
Epoch 100/100
59500/59500 [==============================] - 1s 10us/sample - loss: 0.0201 - val_loss: 0.0214
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Testing Accuracy using Logistic Regression after cleaning: 0.8829
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Testing Accuracy using K nearest neighbors without cleaning: 0.9443
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Process finished with exit code 0
