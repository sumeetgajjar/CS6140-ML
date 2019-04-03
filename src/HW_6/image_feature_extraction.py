import os
from math import floor

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score

from HW_5.AdaBoost import DecisionStumpType
from HW_5.ecoc import ECOC
from HW_6 import utils

BLACK_THRESHOLD = 0
IMAGE_SHAPE = (28, 28)
RECTANGLE_COUNT = 200


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


def find_no_of_black_pixels_along_diagonal_sub_rect(image):
    black_pixels = np.zeros(image.shape)
    n = image.shape[0]

    black_pixels[0, 0] = 1 if image[0, 0] > BLACK_THRESHOLD else 0
    for i in range(1, n):
        black_pixels[0, i] = black_pixels[0, i - 1] + (1 if image[0, i] > BLACK_THRESHOLD else 0)

    for j in range(1, n):
        black_pixels[j, 0] = black_pixels[j - 1, 0] + (1 if image[j, 0] > BLACK_THRESHOLD else 0)

    for i in range(1, n):
        for j in range(1, n):
            current = 1 if image[i, j] > BLACK_THRESHOLD else 0
            black_pixels[i, j] = black_pixels[i, j - 1] + black_pixels[i - 1, j] - black_pixels[i - 1, j - 1] + current

    return black_pixels


def sample_sub_rectangles(n, width, height):
    start_x = np.random.randint(0, width - 8, n)
    start_y = np.random.randint(0, height - 8, n)

    end_x = np.zeros(n, dtype=np.int)
    end_y = np.zeros(n, dtype=np.int)
    for i in range(start_x.shape[0]):
        end_x[i] = np.random.randint(start_x[i], width)
        end_y[i] = np.random.randint(start_y[i], height)

    return np.column_stack((start_x, start_y, end_x, end_y))


def compute_black_pixels_count(black_pixels, start_x, start_y, end_x, end_y):
    count = black_pixels[end_y, end_x]

    if start_y != 0:
        count -= black_pixels[start_y - 1, end_x]

    if start_x != 0:
        count -= black_pixels[end_y, start_x - 1]

    if start_x != 0 and start_y != 0:
        count += black_pixels[start_y - 1, start_x - 1]

    return count


def compute_haar_feature(image, rects):
    image = image.reshape(IMAGE_SHAPE)
    black_pixels = find_no_of_black_pixels_along_diagonal_sub_rect(image)

    feature = np.ones(2 * rects.shape[0])
    for i in range(rects.shape[0]):
        a_x, a_y, d_x, d_y = rects[i]
        b_x, b_y = d_x, a_y
        c_x, c_y = a_x, d_y
        q_x, q_y = a_x, int((a_y + c_y) / 2)
        r_x, r_y = d_x, int((b_y + d_y) / 2)
        m_x, m_y = int((a_x + b_x) / 2), a_y
        n_x, n_y = int((c_x + d_x) / 2), d_y

        b_abqr = compute_black_pixels_count(black_pixels, a_x, a_y, r_x, r_y)
        b_qrcd = compute_black_pixels_count(black_pixels, q_x, q_y, d_x, d_y)
        b_amcn = compute_black_pixels_count(black_pixels, a_x, a_y, n_x, n_y)
        b_mbnd = compute_black_pixels_count(black_pixels, m_x, m_y, d_x, d_y)

        feature[2 * i] = b_abqr - b_qrcd
        feature[(2 * i) + 1] = b_amcn - b_mbnd

    return feature


def compute_haar_feature_wrapper(args):
    image, rects = args
    return compute_haar_feature(image, rects)


def extract_features_from_images(data):
    rects = sample_sub_rectangles(RECTANGLE_COUNT, IMAGE_SHAPE[0], IMAGE_SHAPE[1])

    for s in ['training', 'testing']:
        images = data[s]['images']

        features = Parallel(n_jobs=12, verbose=1, backend="threading")(
            map(delayed(compute_haar_feature_wrapper), [(image, rects) for image in images]))

        data[s]['features'] = np.array(features)

    return data


def get_mnist_images_features(percentage=100, overwrite=False):
    cache_dir = '{}data/mnist/cache/'.format(utils.ROOT)

    if overwrite or not os.path.exists('{}__{}.npy'.format(cache_dir, percentage)):
        data = utils.get_mnist_data()
        data = sample_data(data, percentage)
        data = extract_features_from_images(data)

        np.save('{}{}_training_features.npy'.format(cache_dir, percentage), data['training']['features'])
        np.save('{}{}_training_labels.npy'.format(cache_dir, percentage), data['training']['labels'])
        np.save('{}{}_testing_features.npy'.format(cache_dir, percentage), data['testing']['features'])
        np.save('{}{}_testing_labels.npy'.format(cache_dir, percentage), data['testing']['labels'])
        np.save('{}__{}.npy'.format(cache_dir, percentage), np.empty(0))

        del data['training']['images']
        del data['testing']['images']

        return data
    else:
        training_features = np.load('{}{}_training_features.npy'.format(cache_dir, percentage))
        training_labels = np.load('{}{}_training_labels.npy'.format(cache_dir, percentage))
        testing_features = np.load('{}{}_testing_features.npy'.format(cache_dir, percentage))
        testing_labels = np.load('{}{}_testing_labels.npy'.format(cache_dir, percentage))

        return {
            'training': {
                'features': training_features,
                'labels': training_labels
            },
            'testing': {
                'features': testing_features,
                'labels': testing_labels
            }
        }


def demo_haar_feature_extraction_on_mnist_data():
    data = get_mnist_images_features(percentage=20, overwrite=False)

    training_features = data['training']['features']
    training_labels = data['training']['labels']

    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    classifier = ECOC(training_features, training_labels, testing_features, testing_labels, 50, 10, 50,
                      DecisionStumpType.OPTIMAL, display=True, display_step=1)

    predicted_labels = classifier.predict(training_features)
    print("Training Accuracy:", accuracy_score(training_labels, predicted_labels))

    predicted_labels = classifier.predict(testing_features)
    print("Testing Accuracy:", accuracy_score(testing_labels, predicted_labels))


if __name__ == '__main__':
    np.random.seed(11)
    demo_haar_feature_extraction_on_mnist_data()
