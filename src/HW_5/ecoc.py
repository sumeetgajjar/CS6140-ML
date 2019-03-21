import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score

from HW_5 import utils
from HW_5.AdaBoost import AdaBoost, DecisionStumpType
from HW_5.utils import NewsGroupDataParser


class ECOC:

    def __init__(self, training_features, training_labels, testing_features, testing_labels) -> None:
        super().__init__()
        self.code_label_mapping = None
        self.thresholds = None
        self.classifiers = []
        self.__train(training_features, training_labels, testing_features, testing_labels)

    def __train(self, training_features, training_labels, testing_features, testing_labels):
        print("+" * 40, "Converting the labels into codes", "+" * 40)
        no_of_bits, training_codes, testing_codes, code_label_mapping = self.__convert_labels_to_codes(training_labels,
                                                                                                       testing_labels,
                                                                                                       8)
        # for i in range(no_of_bits):
        #     print("+" * 40, "Training Classifier :", i + 1, "+" * 40)
        #     classifier = self.__train_classifier_for_one_bit(training_features, training_codes[:, i],
        #                                                      testing_features, testing_codes[:, i])
        #     self.classifiers.append(classifier)

        arg_list = [(training_features, training_codes[:, i], testing_features, testing_codes[:, i])
                    for i in range(no_of_bits)]

        self.classifiers = Parallel(n_jobs=32, verbose=50, backend="threading")(
            map(delayed(ECOC.__train_classifier_for_one_bit_args), arg_list))

        self.no_of_bits = no_of_bits
        self.code_label_mapping = code_label_mapping
        self.thresholds = self.__get_thresholds(testing_features, testing_codes)

    @staticmethod
    def __train_classifier_for_one_bit_args(args):
        training_features, training_labels, testing_features, testing_labels = args
        return ECOC.__train_classifier_for_one_bit(training_features, training_labels, testing_features, testing_labels)

    @staticmethod
    def __train_classifier_for_one_bit(training_features, training_labels, testing_features, testing_labels):
        classifier = AdaBoost(DecisionStumpType.RANDOM)
        classifier.train(training_features, training_labels, testing_features, testing_labels, 2500, 1000,
                         display=False)
        return classifier

    def __get_thresholds(self, features, codes):
        thresholds = []

        for i in range(self.no_of_bits):
            prediction = self.classifiers[i].predict(features)
            acc, labels, thr = utils.convert_predictions_to_labels(codes[:, i], prediction)
            thresholds.append(thr)

            print("Classifier:{}, Accuracy:{}, Threshold:{}", i, acc, thr)
            i += 1

        return np.array(thresholds)

    def predict(self, features):

        i = 0
        predicted_codes = []
        for classifier in self.classifiers:
            prediction = classifier.predict(features)
            code = np.ones(features.shape[0])
            code[prediction <= self.thresholds[i]] = -1

            predicted_codes.append(code)

        predicted_codes = np.transpose(predicted_codes)

        min_distance = np.ones(features.shape[0]) * self.no_of_bits
        predicted_labels = np.ones(features.shape[0]) * -1
        for (code, label) in self.code_label_mapping:
            distance = np.count_nonzero(np.subtract(predicted_codes, code), axis=1)
            min_distance_mask = distance < min_distance

            predicted_labels[min_distance_mask] = label
            min_distance[min_distance_mask] = distance[min_distance_mask]

        return predicted_labels

    def __convert_labels_to_codes(self, training_labels, testing_labels, no_of_classes):
        labels = np.unique(training_labels)
        no_of_bits, codes = self.generate_ecoc_exhaustive_code(no_of_classes)

        training_codes = np.ones((training_labels.shape[0], no_of_bits))
        testing_codes = np.ones((testing_labels.shape[0], no_of_bits))
        code_label_mapping = []
        for label, code in zip(labels, codes):
            training_codes[training_labels == label] *= code
            testing_codes[testing_labels == label] *= code
            code_label_mapping.append((code, label))

        return no_of_bits, training_codes, testing_codes, code_label_mapping

    @staticmethod
    def generate_ecoc_exhaustive_code(no_of_classes):
        no_of_bits = 2 ** (no_of_classes - 1)

        codes = [[1 for _ in range(no_of_bits)]]
        for i in range(2, no_of_classes + 1):
            no_of_ones = 2 ** (no_of_classes - i)
            no_of_zeros = 2 ** (no_of_classes - i)

            code = []
            for j in range(2 ** (i - 2)):
                code.extend([-1] * no_of_zeros)
                code.extend([1] * no_of_ones)

            codes.append(code)

        no_of_bits -= 1
        codes = np.array(codes)
        codes = codes[:, :-1].copy()
        return no_of_bits, codes


def demo_ecoc_on_8_news_group_data():
    print("+" * 40, "Parsing Data", "+" * 40)
    data = NewsGroupDataParser().parse_data()
    training_features = data['training']['features']
    training_labels = data['training']['labels']
    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    classifier = ECOC(training_features, training_labels, testing_features, testing_labels)
    predicted_labels = classifier.predict(testing_features)
    print("Testing Accuracy:", accuracy_score(testing_labels, predicted_labels))


if __name__ == '__main__':
    demo_ecoc_on_8_news_group_data()
