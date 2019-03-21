import numpy as np

from HW_5 import utils
from HW_5.AdaBoost import AdaBoost, DecisionStumpType


class ActiveLearning:

    @staticmethod
    def __random_sample_data(features, labels, N):
        random_indexes = np.random.random_integers(0, features.shape[0] - 1, N)
        mask = np.zeros(features.shape[0], dtype=np.bool)
        mask[random_indexes] = True

        return features[mask], labels[mask], features[~mask], labels[~mask]

    @staticmethod
    def __get_x_percentage_of_data(features, labels, x):
        N = np.ceil((x / 100) * features.shape[0]).astype(np.int)
        return ActiveLearning.__random_sample_data(features, labels, N)

    def select_closet_points_to_separation(self, adaboost, features, labels, threshold, c):
        predictions = adaboost.predict(features)
        sorted_indices = np.argsort(np.abs(predictions - threshold))
        N = np.ceil(np.ceil((c / 100) * features.shape[0])).astype(np.int)

        subset = sorted_indices[:N]
        mask = np.zeros(features.shape[0], dtype=np.bool)
        mask[subset] = True

        return features[mask], labels[mask], features[~mask], labels[~mask]

    def train(self, training_features, training_labels, testing_features, testing_labels):
        t_f_s, t_l_s, t_f_r, t_l_r = self.__get_x_percentage_of_data(training_features, training_labels, 5)
        temp_training_features = t_f_s.copy()
        temp_training_labels = t_l_s.copy()

        i = 1
        training_data_percentage = 5
        while training_data_percentage < 50:
            adaboost = AdaBoost(DecisionStumpType.OPTIMAL)
            adaboost.train(temp_training_features, temp_training_labels, testing_features, testing_labels, 10,
                           display=False)

            training_features_predictions = adaboost.predict(temp_training_features)
            training_acc, labels, thr = utils.convert_predictions_to_labels(temp_training_labels,
                                                                            training_features_predictions)

            testing_features_predictions = adaboost.predict(testing_features)
            testing_acc, labels, thr = utils.convert_predictions_to_labels(testing_labels,
                                                                           testing_features_predictions)

            print("+" * 80)
            print("Round:{}, Training Data:{}%, Training Acc:{}, Testing Acc:{}".format(i, training_data_percentage,
                                                                                        training_acc, testing_acc))
            print("+" * 80)

            t_f_s, t_l_s, t_f_r, t_l_r = self.select_closet_points_to_separation(adaboost, t_f_r, t_l_r, thr, 2)

            temp_training_features = np.append(temp_training_features, t_f_s, axis=0)
            temp_training_labels = np.append(temp_training_labels, t_l_s, axis=0)
            training_data_percentage = (temp_training_features.shape[0] / training_features.shape[0]) * 100
            i += 1


def demo_active_learning_on_spam():
    data = utils.get_spam_data_for_ada_boost()
    classifier = ActiveLearning()

    data = utils.k_fold_split(10, data, seed=11, shuffle=True)[0]
    training_features = data['training']['features']
    training_labels = data['training']['labels']

    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    classifier.train(training_features, training_labels, testing_features, testing_labels)


if __name__ == '__main__':
    demo_active_learning_on_spam()
