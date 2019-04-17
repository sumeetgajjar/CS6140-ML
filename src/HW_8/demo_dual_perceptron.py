import numpy as np
from sklearn.metrics import accuracy_score

from HW_8 import utils
from HW_8.dual_perceptron import DualPerceptron
from HW_8.knn import SimilarityMeasures


def demo_perceptron():
    print("+" * 40, "Dual Perceptron demo on normal perceptron data", "+" * 40)
    data = utils.get_perceptron_data()
    perceptron = DualPerceptron(0.02, SimilarityMeasures.dot_product)
    features = data['features']
    features = utils.normalize_data_using_zero_mean_unit_variance(features)
    features = utils.prepend_one_to_feature_vectors(features)

    labels = data['labels']
    perceptron.train(features, labels)
    print("M Counts: ", perceptron.m.tolist())

    predicted_labels = perceptron.predict(features)
    print("Accuracy: ", accuracy_score(labels, predicted_labels))
    print("+" * 40, "Dual Perceptron demo on normal perceptron data", "+" * 40)
    print()


if __name__ == '__main__':
    np.random.seed(11)
    demo_perceptron()
