from math import floor

import numpy as np

from HW_6 import utils


def sample_data(data, percentage):
    for s in ['training', 'testing']:
        images = data[s]['images']
        labels = data[s]['labels']

        sampled_images = np.empty((0, images.shape[1]))
        sampled_labels = np.empty(0)
        for i in range(10):
            digit_i_images = images[labels == i]
            digit_i_labels = labels[labels == i]

            total_size = digit_i_labels.shape[0]
            sampled_size = floor((percentage / 100) * total_size)

            indexes = np.arange(sampled_size)
            np.random.shuffle(indexes)

            sampled_images = np.append(sampled_images, digit_i_images[indexes[:sampled_size]], axis=0)
            sampled_labels = np.append(sampled_labels, digit_i_labels[indexes[:sampled_size]])

        data[s]['images'] = sampled_images
        data[s]['labels'] = sampled_labels

    return data


def demo_haar_feature_extraction_on_mnist_data():
    data = utils.get_mnist_data()
    data = sample_data(data, 20)


if __name__ == '__main__':
    demo_haar_feature_extraction_on_mnist_data()
