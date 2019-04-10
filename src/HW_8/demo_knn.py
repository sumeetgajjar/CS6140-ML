from sklearn.metrics import accuracy_score

from HW_6.image_feature_extraction import get_mnist_images_features
from HW_8 import utils
from HW_8.knn import KNN, Kernel


def demo_wrapper(data, kernel, k_list=(1, 3, 7), n_jobs=1, verbose=0):
    training_features = data['training']['features']
    training_labels = data['training']['labels']
    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    classifier = KNN(kernel, training_features, training_labels, n_jobs=n_jobs, verbose=verbose)

    pred_training_labels = classifier.predict(training_features, k_list)
    pred_testing_labels = classifier.predict(testing_features, k_list)

    for ix, k in enumerate(k_list):
        training_acc = accuracy_score(training_labels, pred_training_labels[:, ix])
        print("k=>{}, Training Accuracy:{}".format(k, training_acc))
        testing_acc = accuracy_score(testing_labels, pred_testing_labels[:, ix])
        print("k=>{}, Testing Accuracy:{}".format(k, testing_acc))
        print()


def demo_knn_on_spam_base_data():
    print("+" * 40, "KNN on Spambase Data with Euclidean Kernel", "+" * 40)
    data = utils.get_spam_data()
    data = utils.k_fold_split(10, data, 11, shuffle=True)[0]
    demo_wrapper(data, Kernel.EUCLIDEAN)
    print("+" * 40, "KNN on Spambase Data with Euclidean Kernel", "+" * 40)


def demo_knn_on_mnist_data():
    for kernel in (Kernel.COSINE, Kernel.GAUSSIAN, Kernel.POLYNOMIAL):
        print("+" * 40, "KNN on MNIST Data with {} Kernel".format(kernel), "+" * 40)
        data = get_mnist_images_features(percentage=20)
        demo_wrapper(data, kernel, n_jobs=10, verbose=1)
        print("+" * 40, "KNN on MNIST Data with {} Kernel".format(kernel), "+" * 40)
        print()


if __name__ == '__main__':
    # demo_knn_on_spam_base_data()
    demo_knn_on_mnist_data()
