from sklearn.metrics import accuracy_score
from sklearn.svm import NuSVC

from HW_7 import utils


def demo_classifier(classifier):
    data = utils.get_spam_data()
    data = utils.k_fold_split(10, data, seed=11, shuffle=True)[0]

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


def demo_sklearn_rbf_svm_on_spam():
    print("+" * 40, "Sklearn SVM RBF", "+" * 40)
    classifier = NuSVC(kernel='rbf', gamma='scale', random_state=11)
    demo_classifier(classifier)
    print("+" * 40, "Sklearn SVM RBF", "+" * 40)
    print()


def demo_sklearn_linear_svm_on_spam():
    print("+" * 40, "Sklearn SVM Linear", "+" * 40)
    classifier = NuSVC(kernel='linear', gamma='scale', random_state=11)
    demo_classifier(classifier)
    print("+" * 40, "Sklearn SVM Linear", "+" * 40)
    print()


def demo_sklearn_poly_svm_on_spam():
    print("+" * 40, "Sklearn SVM Poly", "+" * 40)
    classifier = NuSVC(kernel='poly', gamma='scale', degree=2, random_state=11)
    demo_classifier(classifier)
    print("+" * 40, "Sklearn SVM Poly", "+" * 40)
    print()


def demo_sklearn_sigmoid_svm_on_spam():
    print("+" * 40, "Sklearn SVM Sigmoid", "+" * 40)
    classifier = NuSVC(kernel='sigmoid', gamma='scale', random_state=11)
    demo_classifier(classifier)
    print("+" * 40, "Sklearn SVM Sigmoid", "+" * 40)
    print()


if __name__ == '__main__':
    demo_sklearn_rbf_svm_on_spam()
    demo_sklearn_linear_svm_on_spam()
    demo_sklearn_poly_svm_on_spam()
    demo_sklearn_sigmoid_svm_on_spam()
