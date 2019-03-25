import numpy as np
from sklearn.metrics import accuracy_score

from HW_4.naive_bayes import NaiveBayesBins
from HW_6 import utils


def demo_naive_bayes_on_spam_polluted():
    print("+" * 40, "Naive Bayes", "+" * 40)
    data = utils.get_spam_polluted_data()

    training_features = data['training']['features']
    training_labels = data['training']['labels']

    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    combined_features = np.concatenate((training_features, testing_features))
    combined_labels = np.concatenate((training_labels, testing_labels))

    converted_features = NaiveBayesBins.convert_continuous_features_to_four_bins(combined_features, combined_labels)
    training_features = converted_features[:training_features.shape[0]]
    testing_features = converted_features[training_features.shape[0]:]

    classifier = NaiveBayesBins(1, 4, training_features.shape[1], 1)
    classifier.train(training_features, training_labels)

    training_predictions = classifier.predict(training_features)
    training_prediction_labels = np.argmax(training_predictions, axis=1)
    training_accuracy = accuracy_score(training_labels, training_prediction_labels)

    testing_predictions = classifier.predict(testing_features)
    testing_prediction_labels = np.argmax(testing_predictions, axis=1)
    testing_accuracy = accuracy_score(testing_labels, testing_prediction_labels)

    print("Training Accuracy", training_accuracy)
    print("Testing Accuracy", testing_accuracy)
    print("+" * 40, "Naive Bayes", "+" * 40)


if __name__ == '__main__':
    demo_naive_bayes_on_spam_polluted()
