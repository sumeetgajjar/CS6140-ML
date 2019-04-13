from sklearn.metrics import accuracy_score

from HW_6.image_feature_extraction import get_mnist_images_features
from HW_8 import utils
from HW_8.knn import KNN, Kernel, KNNMode


def demo_wrapper(data, kernel, mode=KNNMode.K_POINTS, k_list=(1, 3, 7), n_jobs=1, verbose=0):
    training_features = data['training']['features']
    training_labels = data['training']['labels']
    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    classifier = KNN(kernel, training_features, training_labels, mode=mode, n_jobs=n_jobs, verbose=verbose)

    pred_training_labels = classifier.predict(training_features, k_list)
    pred_testing_labels = classifier.predict(testing_features, k_list)

    for ix, k in enumerate(k_list):
        training_acc = accuracy_score(training_labels, pred_training_labels[:, ix])
        print("{}=>{}, Training Accuracy:{}".format(mode.get_str(), k, training_acc))
        testing_acc = accuracy_score(testing_labels, pred_testing_labels[:, ix])
        print("{}=>{}, Testing Accuracy:{}".format(mode.get_str(), k, testing_acc))
        print()


def demo_k_points_knn_on_spam_base_data():
    print("+" * 40, "K Points KNN on Spambase Data with Euclidean Kernel", "+" * 40)
    data = utils.get_spam_data()
    data['features'] = utils.normalize_data_using_zero_mean_unit_variance(data['features'])
    data = utils.k_fold_split(10, data, 11, shuffle=True)[0]

    demo_wrapper(data, Kernel.EUCLIDEAN, k_list=[1, 2, 3])
    print("+" * 40, "K Points KNN on Spambase Data with Euclidean Kernel", "+" * 40)


def demo_window_knn_on_spam_base_data():
    print("+" * 40, "Window KNN on Spambase Data with Euclidean Kernel", "+" * 40)
    data = utils.get_spam_data()
    data['features'] = utils.normalize_data_using_zero_mean_unit_variance(data['features'])
    data = utils.k_fold_split(10, data, 11, shuffle=True)[0]

    demo_wrapper(data, Kernel.EUCLIDEAN, mode=KNNMode.RADIUS, k_list=[1, 2.5])
    print("+" * 40, "Window KNN on Spambase Data with Euclidean Kernel", "+" * 40)


def demo_k_points_knn_on_mnist_data():
    for kernel in (Kernel.COSINE, Kernel.GAUSSIAN, Kernel.POLYNOMIAL):
        print("+" * 40, "K Points KNN on MNIST Data with {} Kernel".format(kernel), "+" * 40)
        data = get_mnist_images_features(percentage=20)
        demo_wrapper(data, kernel, n_jobs=10, verbose=1)
        print("+" * 40, "K Points KNN on MNIST Data with {} Kernel".format(kernel), "+" * 40)
        print()


def demo_window_knn_on_mnist_data():
    kernel = Kernel.COSINE
    print("+" * 40, "WINDOW KNN on MNIST Data with {} Kernel".format(kernel), "+" * 40)
    data = get_mnist_images_features(percentage=20)
    demo_wrapper(data, kernel, mode=KNNMode.RADIUS, k_list=[0.83], n_jobs=10, verbose=1)
    print("+" * 40, "WINDOW KNN on MNIST Data with {} Kernel".format(kernel), "+" * 40)
    print()


if __name__ == '__main__':
    # demo_k_points_knn_on_spam_base_data()
    # demo_k_points_knn_on_mnist_data()
    # demo_window_knn_on_spam_base_data()
    demo_window_knn_on_mnist_data()
