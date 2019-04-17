import numpy as np
from sklearn.metrics import accuracy_score

from HW_6.image_feature_extraction import get_mnist_images_features
from HW_8 import utils
from HW_8.knn import KNN, Kernel, KNNMode, KNNUsingKernelDensity, SimilarityMeasures


def demo_wrapper(data, kernel, mode=KNNMode.K_POINTS, k_list=(1, 3, 7), n_jobs=1, verbose=0):
    features = data['features']
    labels = data['labels']

    classifier = KNN(kernel, mode=mode, n_jobs=n_jobs, verbose=verbose)
    pred_labels = classifier.predict(features, labels, k_list)

    for ix, k in enumerate(k_list):
        acc = accuracy_score(labels, pred_labels[:, ix])
        print("{}=>{}, Training Accuracy:{}".format(mode.get_str(), k, acc))
        print()


def demo_k_points_knn_on_spam_base_data():
    print("+" * 40, "K Points KNN on Spambase Data with Euclidean Kernel", "+" * 40)
    data = utils.get_spam_data()
    data['features'] = utils.normalize_data_using_zero_mean_unit_variance(data['features'])
    # data = utils.k_fold_split(10, data, 11, shuffle=True)[0]

    demo_wrapper(data, Kernel.EUCLIDEAN)
    print("+" * 40, "K Points KNN on Spambase Data with Euclidean Kernel", "+" * 40)
    print()


def demo_window_knn_on_spam_base_data():
    print("+" * 40, "Window KNN on Spambase Data with Euclidean Kernel", "+" * 40)
    data = utils.get_spam_data()
    data['features'] = utils.normalize_data_using_zero_mean_unit_variance(data['features'])
    demo_wrapper(data, Kernel.EUCLIDEAN, mode=KNNMode.RADIUS, k_list=[2.5])
    print("+" * 40, "Window KNN on Spambase Data with Euclidean Kernel", "+" * 40)
    print()


def demo_k_points_knn_on_mnist_data():
    for kernel in [Kernel.COSINE, Kernel.GAUSSIAN, Kernel.POLYNOMIAL]:
        print("+" * 40, "K Points KNN on MNIST Data with {} Kernel".format(kernel), "+" * 40)
        data = get_mnist_images_features(percentage=20)

        features = np.append(data['training']['features'], data['testing']['features'], axis=0)
        labels = np.append(data['training']['labels'], data['testing']['labels'], axis=0)
        _data = dict({'features': features, 'labels': labels})

        demo_wrapper(_data, kernel, n_jobs=10, verbose=1)
        print("+" * 40, "K Points KNN on MNIST Data with {} Kernel".format(kernel), "+" * 40)
        print()


def demo_window_knn_on_mnist_data():
    kernel = Kernel.COSINE
    print("+" * 40, "WINDOW KNN on MNIST Data with {} Kernel".format(kernel), "+" * 40)
    data = get_mnist_images_features(percentage=10)

    features = np.append(data['training']['features'], data['testing']['features'], axis=0)
    labels = np.append(data['training']['labels'], data['testing']['labels'])
    _data = dict({'features': features, 'labels': labels})

    demo_wrapper(_data, kernel, mode=KNNMode.RADIUS, k_list=[0.9, 0.93, 0.95], n_jobs=10,
                 verbose=1)
    print("+" * 40, "WINDOW KNN on MNIST Data with {} Kernel".format(kernel), "+" * 40)
    print()


def demo_kernel_density_knn_on_spam_base_data():
    print("+" * 40, "Kernel Density KNN on Spambase Data with Gaussian Kernel", "+" * 40)
    data = utils.get_spam_data()
    data['features'] = utils.normalize_data_using_zero_mean_unit_variance(data['features'])

    classifier = KNNUsingKernelDensity(SimilarityMeasures.gaussian)
    predictions = classifier.predict(data['features'], data['labels'])
    print("Training Accuracy:{}".format(accuracy_score(data['labels'], predictions)))

    print("+" * 40, "Kernel Density KNN on Spambase Data with Gaussian Kernel", "+" * 40)
    print()


def demo_kernel_density_knn_on_mnist_data():
    print("+" * 40, "Kernel Density KNN on MNIST Data with Gaussian Kernel", "+" * 40)
    data = get_mnist_images_features(percentage=10)

    features = np.append(data['training']['features'], data['testing']['features'], axis=0)
    labels = np.append(data['training']['labels'], data['testing']['labels'])
    _data = dict({'features': features, 'labels': labels})

    classifier = KNNUsingKernelDensity(SimilarityMeasures.gaussian, n_jobs=10)
    predictions = classifier.predict(_data['features'], _data['labels'])

    print("Training Accuracy:{}".format(accuracy_score(_data['labels'], predictions)))

    print("+" * 40, "Kernel Density KNN on MNIST Data with Gaussian Kernel", "+" * 40)
    print()

    print("+" * 40, "Kernel Density KNN on MNIST Data with Gaussian Kernel", "+" * 40)

    classifier = KNNUsingKernelDensity(SimilarityMeasures.polynomial, n_jobs=10)
    predictions = classifier.predict(_data['features'], _data['labels'])
    print("Training Accuracy:{}".format(accuracy_score(_data['labels'], predictions)))

    print("+" * 40, "Kernel Density KNN on MNIST Data with Gaussian Kernel", "+" * 40)
    print()


if __name__ == '__main__':
    np.random.seed(1)
    # demo_k_points_knn_on_spam_base_data()
    # demo_window_knn_on_spam_base_data()
    # demo_k_points_knn_on_mnist_data()
    # demo_window_knn_on_mnist_data()
    # demo_kernel_density_knn_on_spam_base_data()
    # sigma = 4
    demo_kernel_density_knn_on_mnist_data()
