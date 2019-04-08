import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from HW_6.image_feature_extraction import get_mnist_images_features
from HW_7 import utils


def demo_classifier(data, classifier):
    training_features = data['training']['features']
    training_labels = data['training']['labels']

    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    classifier.fit(training_features, training_labels)

    training_prediction_labels = classifier.predict(training_features)
    training_accuracy = accuracy_score(training_labels, training_prediction_labels)

    testing_prediction_labels = classifier.predict(testing_features)
    testing_accuracy = accuracy_score(testing_labels, testing_prediction_labels)

    print("Training Accuracy", training_accuracy)
    print("Testing Accuracy", testing_accuracy)


def demo_classifier_wrapper(classifier):
    data = utils.get_spam_data()
    data = utils.k_fold_split(10, data, seed=11, shuffle=True)[0]

    training_features = data['training']['features']
    testing_features = data['testing']['features']

    combined_features = np.concatenate((training_features, testing_features))
    normalized_features = utils.normalize_data_using_zero_mean_unit_variance(combined_features)

    training_features = normalized_features[:training_features.shape[0]]
    testing_features = normalized_features[training_features.shape[0]:]

    data['training']['features'] = training_features
    data['testing']['features'] = testing_features

    demo_classifier(data, classifier)


def demo_sklearn_rbf_svm_on_spam():
    print("+" * 40, "Sklearn SVM RBF", "+" * 40)
    classifier = SVC(C=1, kernel='rbf', gamma='scale', random_state=11)
    demo_classifier_wrapper(classifier)
    print("+" * 40, "Sklearn SVM RBF", "+" * 40)
    print()


def demo_sklearn_linear_svm_on_spam():
    print("+" * 40, "Sklearn SVM Linear", "+" * 40)
    classifier = SVC(C=1, kernel='linear', gamma='scale', random_state=11)
    demo_classifier_wrapper(classifier)
    print("+" * 40, "Sklearn SVM Linear", "+" * 40)
    print()


def demo_sklearn_poly_svm_on_spam():
    print("+" * 40, "Sklearn SVM Poly", "+" * 40)
    classifier = SVC(kernel='poly', gamma='scale', degree=2, random_state=11)
    demo_classifier_wrapper(classifier)
    print("+" * 40, "Sklearn SVM Poly", "+" * 40)
    print()


def demo_sklearn_sigmoid_svm_on_spam():
    print("+" * 40, "Sklearn SVM Sigmoid", "+" * 40)
    classifier = SVC(C=1, kernel='sigmoid', gamma='scale', random_state=11)
    demo_classifier_wrapper(classifier)
    print("+" * 40, "Sklearn SVM Sigmoid", "+" * 40)
    print()


def demo_svm_on_mnist():
    print("+" * 40, "Sklearn Multi class SVM", "+" * 40)
    data = get_mnist_images_features(percentage=20)

    classifier = SVC(C=1, kernel='poly', gamma='scale', degree=1)

    demo_classifier(data, classifier)
    print("+" * 40, "Sklearn Multi class SVM", "+" * 40)


if __name__ == '__main__':
    demo_sklearn_rbf_svm_on_spam()
    demo_sklearn_linear_svm_on_spam()
    demo_sklearn_poly_svm_on_spam()
    demo_sklearn_sigmoid_svm_on_spam()
    demo_svm_on_mnist()
