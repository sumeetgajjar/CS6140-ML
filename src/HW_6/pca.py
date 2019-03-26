import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from HW_4.naive_bayes import NaiveBayesGaussian
from HW_6 import utils


def demo_classifier(data):
    training_features = data['training']['features']
    training_labels = data['training']['labels']

    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    classifier = NaiveBayesGaussian(0.7)
    classifier.train(training_features, training_labels)

    training_predictions = classifier.predict(training_features)
    training_prediction_labels = np.argmax(training_predictions, axis=1)
    training_accuracy = accuracy_score(training_labels, training_prediction_labels)

    testing_predictions = classifier.predict(testing_features)
    testing_prediction_labels = np.argmax(testing_predictions, axis=1)
    testing_accuracy = accuracy_score(testing_labels, testing_prediction_labels)

    print("Training Accuracy", training_accuracy)
    print("Testing Accuracy", testing_accuracy)


def demo_naive_bayes_on_spam_polluted_pca():
    print("+" * 40, "Naive Bayes After PCA", "+" * 40)
    data = utils.get_spam_polluted_data()

    training_features = data['training']['features']
    testing_features = data['testing']['features']

    combined_features = np.concatenate((training_features, testing_features))

    pca = PCA(n_components=100)
    pca.fit(combined_features)
    transformed_features = pca.transform(combined_features)

    training_features = transformed_features[:training_features.shape[0]]
    testing_features = transformed_features[training_features.shape[0]:]

    data['training']['features'] = training_features
    data['testing']['features'] = testing_features

    demo_classifier(data)
    print("+" * 40, "Naive Bayes After PCA", "+" * 40)


def demo_naive_bayes_on_spam_polluted():
    print("+" * 40, "Naive Bayes before PCA", "+" * 40)
    data = utils.get_spam_polluted_data()
    demo_classifier(data)
    print("+" * 40, "Naive Bayes before PCA", "+" * 40)


if __name__ == '__main__':
    demo_naive_bayes_on_spam_polluted()
    demo_naive_bayes_on_spam_polluted_pca()
