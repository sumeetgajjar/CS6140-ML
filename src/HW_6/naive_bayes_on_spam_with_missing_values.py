import numpy as np
from sklearn.metrics import accuracy_score

from HW_4.naive_bayes import NaiveBayesBins
from HW_6 import utils


def demo_naive_bayes_bernoulli_on_spam_with_missing_data():
    print("+" * 40, "Bernoulli Naive Bayes with Missing Data in spam", "+" * 40)
    data = utils.get_spam_missing_data()

    training_features = data['training']['features']
    training_labels = data['training']['labels']
    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    combined_features = np.concatenate((training_features, testing_features))
    converted_features = NaiveBayesBins.convert_continuous_features_to_discrete(combined_features)
    training_features = converted_features[:training_features.shape[0]]
    testing_features = converted_features[training_features.shape[0]:]

    classifier = NaiveBayesBins(1, 2, training_features.shape[1], 1)
    classifier.train(training_features, training_labels)

    training_predictions = classifier.predict(training_features)
    training_prediction_labels = np.argmax(training_predictions, axis=1)

    testing_predictions = classifier.predict(testing_features)
    testing_prediction_labels = np.argmax(testing_predictions, axis=1)

    print("Training Accuracy:", accuracy_score(training_labels, training_prediction_labels))
    print("Testing Accuracy:", accuracy_score(testing_labels, testing_prediction_labels))


if __name__ == '__main__':
    demo_naive_bayes_bernoulli_on_spam_with_missing_data()
