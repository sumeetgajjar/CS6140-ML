from sklearn.svm import SVC

from HW_6.image_feature_extraction import get_mnist_images_features
from HW_7.svm_on_spam import demo_classifier


def demo_svm_on_mnist():
    print("+" * 40, "Sklearn Multi class SVM", "+" * 40)
    data = get_mnist_images_features(percentage=100, overwrite=True)

    classifier = SVC(C=1, kernel='poly', gamma='scale', degree=2)

    demo_classifier(data, classifier)
    print("+" * 40, "Sklearn Multi class SVM", "+" * 40)


if __name__ == '__main__':
    demo_svm_on_mnist()
