import numpy as np
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.metrics import accuracy_score

from HW_4.naive_bayes import NaiveBayesGaussian
from HW_6 import utils


def demo_classifier(data):
    training_features = data['training']['features']
    training_labels = data['training']['labels']

    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    classifier = NaiveBayesGaussian(0.1)
    classifier.train(training_features, training_labels)

    training_predictions = classifier.predict(training_features)
    training_prediction_labels = np.argmax(training_predictions, axis=1)
    training_accuracy = accuracy_score(training_labels, training_prediction_labels)

    testing_predictions = classifier.predict(testing_features)
    testing_prediction_labels = np.argmax(testing_predictions, axis=1)
    testing_accuracy = accuracy_score(testing_labels, testing_prediction_labels)

    print("Training Accuracy", training_accuracy)
    print("Testing Accuracy", testing_accuracy)


# todo: fix the bug in LDA
def demo_naive_bayes_on_spam_polluted_lda():
    print("+" * 40, "Naive Bayes After LDA", "+" * 40)
    data = utils.get_spam_polluted_data()

    training_features = data['training']['features']
    training_labels = data['training']['labels']
    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    combined_features = np.concatenate((training_features, testing_features))
    combined_labels = np.concatenate((training_labels, testing_labels))

    lda = LatentDirichletAllocation(n_components=40, max_iter=5, random_state=0, n_jobs=4, verbose=10)
    lda.fit(combined_features, combined_labels)
    transformed_features = lda.transform(combined_features)

    training_features = transformed_features[:training_features.shape[0]]
    testing_features = transformed_features[training_features.shape[0]:]

    data['training']['features'] = training_features
    data['testing']['features'] = testing_features

    demo_classifier(data)
    print("+" * 40, "Naive Bayes After LDA", "+" * 40)


def demo_naive_bayes_on_spam_polluted_pca():
    print("+" * 40, "Naive Bayes After PCA", "+" * 40)
    data = utils.get_spam_polluted_data()

    training_features = data['training']['features']
    testing_features = data['testing']['features']

    pca = PCA(n_components=100, random_state=11)
    pca.fit(training_features)
    training_features = pca.transform(training_features)
    testing_features = pca.transform(testing_features)

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
    # demo_naive_bayes_on_spam_polluted()
    # demo_naive_bayes_on_spam_polluted_pca()
    demo_naive_bayes_on_spam_polluted_lda()
