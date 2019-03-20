import numpy as np
import pandas as pd


def get_spam_data():
    data = pd.read_csv('../../data/spam-email/spambase.data', header=None)
    features = np.array(data.iloc[:, 0:57])
    labels = np.array(data.iloc[:, 57])

    if features.shape[0] != labels.shape[0]:
        raise Exception("Mismatch in Feature Tuples(%d) and Label Tuples(%d)" % (features.size, labels.size))

    return {
        'features': features,
        'labels': labels
    }


def normalize_data(data):
    feature_vectors = data['features']

    total_features = len(feature_vectors[0])
    i = 0
    while i < total_features:
        #     for i in [26]:
        feature_values = feature_vectors[:, i]
        mean = feature_values.mean()
        std = feature_values.std()
        normalized_values = (feature_values - mean) / std
        feature_vectors[:, i] = normalized_values

        #         feature_values = feature_vectors[:,i]
        #         f_min = feature_values.min()
        #         f_max = feature_values.max()
        #         normalized_values = (feature_values - f_min)/(f_max - f_min)
        #         feature_vectors[:,i] = normalized_values

        i += 1

    data['features'] = feature_vectors
    return data


def k_fold_split(k, data, shuffle=False):
    sample_size = data['features'].shape[0]

    indices = np.arange(0, sample_size)

    if shuffle:
        np.random.shuffle(indices)

    folds = np.array_split(indices, k)
    testing_fold_index = 0
    final_data = []

    for i in range(0, k):
        training_folds = [folds[j] for j in range(0, k) if j != testing_fold_index]
        training_data_indices = np.concatenate(training_folds)
        training_data_features = data['features'][training_data_indices]
        training_data_labels = data['labels'][training_data_indices]

        testing_data_indices = folds[testing_fold_index]
        testing_data_features = data['features'][testing_data_indices]
        testing_data_labels = data['labels'][testing_data_indices]

        temp = {
            'training': {
                'features': training_data_features,
                'labels': training_data_labels
            },
            'testing': {
                'features': testing_data_features,
                'labels': testing_data_labels
            }
        }

        final_data.append(temp)
        testing_fold_index += 1

    return final_data


class Node(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.splitting_info = None
        self.label_info = None
        self.label = None
        self.entropy = None

    def __str__(self, level=0):
        ret = "{}Label={}, LabelInfo={}, Entropy={}".format("\t" * level, self.label, self.label_info, self.entropy)

        if self.splitting_info is not None:
            ret += " Feature={}, Threshold={}".format(self.splitting_info['splitting_feature_index'],
                                                      self.splitting_info['splitting_threshold'])

        ret += "\n\n"

        if self.left is not None:
            ret += self.left.__str__(level + 1)
        if self.right is not None:
            ret += self.right.__str__(level + 1)

        return ret

    def predict_label(self, feature_vector):
        if self.left is None or self.right is None:
            return self.label

        splitting_threshold = self.splitting_info['splitting_threshold']
        splitting_feature_index = self.splitting_info['splitting_feature_index']
        if feature_vector[splitting_feature_index] <= splitting_threshold:
            return self.left.predict_label(feature_vector)
        else:
            return self.right.predict_label(feature_vector)

    def predict(self, features):
        predictions = []
        for feature in features:
            prediction = self.predict_label(feature)
            predictions.append(prediction)

        return np.array(predictions)


def create_tree_util(current_level, feature_vectors, labels, min_sample_size, max_depth):
    if labels.size < min_sample_size or current_level > max_depth:
        #         print('sample size return')
        return None

    freq = Counter(labels)
    max_freq_label = None
    max_freq = -1
    for k, v in freq.items():
        if max_freq < v:
            max_freq_label = k
            max_freq = v

    root = Node()
    root.label = max_freq_label
    root.label_info = freq
    root.entropy = get_entropy(freq, len(labels))

    splitting_info = get_feature_to_split(labels, feature_vectors)
    if splitting_info is None:
        #         print('splitting info return')
        return root

    splitted_data = split_data(splitting_info, labels, feature_vectors)
    #     print(splitting_info)
    #     print('size', splitted_data['left']['labels'].size, splitted_data['right']['labels'].size)

    if splitted_data['left']['labels'].size == 0 or splitted_data['right']['labels'].size == 0:
        #         print('return size 0')
        return root

    root.splitting_info = splitting_info
    #     print('before left')
    root.left = create_tree_util(current_level + 1, splitted_data['left']['features'], splitted_data['left']['labels'],
                                 min_sample_size, max_depth)
    #     print('before right')
    root.right = create_tree_util(current_level + 1, splitted_data['right']['features'],
                                  splitted_data['right']['labels'], min_sample_size, max_depth)

    return root


def create_tree(feature_vectors, labels, min_sample_size, max_depth):
    return create_tree_util(1, feature_vectors, labels, min_sample_size, max_depth)


def split_data(splitting_info, labels, feature_vectors):
    splitting_threshold = splitting_info['splitting_threshold']
    splitting_feature_index = splitting_info['splitting_feature_index']

    left_part_feature_vectors = []
    left_part_labels = []
    right_part_feature_vectors = []
    right_part_labels = []

    number_of_data_points = len(feature_vectors)

    i = 0
    while i < number_of_data_points:
        if feature_vectors[i][splitting_feature_index] <= splitting_threshold:
            left_part_feature_vectors.append(feature_vectors[i])
            left_part_labels.append(labels[i])
        else:
            right_part_feature_vectors.append(feature_vectors[i])
            right_part_labels.append(labels[i])

        i += 1

    return {
        'left': {
            'features': np.array(left_part_feature_vectors),
            'labels': np.array(left_part_labels)
        },
        'right': {
            'features': np.array(right_part_feature_vectors),
            'labels': np.array(right_part_labels)
        }
    }

    # temp = []
    # i = 0
    # for feature_vector in feature_vectors:
    #     temp.append((i, feature_vector[splitting_feature_index]))
    #     i += 1
    #
    # temp = np.array(sorted(temp, key=lambda item: item[1]))
    # temp = np.array(temp)
    #
    # j = splitting_info['split_tuple_index']
    #
    # left_part = temp[:j + 1, 0].astype(int)
    # right_part = temp[j + 1:, 0].astype(int)
    #
    # return {
    #     'left': {
    #         'features': feature_vectors[left_part],
    #         'labels': labels[left_part]
    #     },
    #     'right': {
    #         'features': feature_vectors[right_part],
    #         'labels': labels[right_part]
    #     }
    # }


from collections import Counter


def get_entropy(freq, number_of_data_points):
    current_entropy = 0
    for label, count in freq.items():
        prob = count / number_of_data_points
        log_prob = np.log2(prob) if prob != 0 else 0
        current_entropy -= (prob * log_prob)

    return current_entropy


def get_feature_to_split(labels, feature_vectors):
    max_information_gain = 0
    splitting_feature_index = None
    splitting_threshold = None
    split_tuple_index = None

    number_of_data_points = len(labels)

    current_entropy = get_entropy(Counter(labels), number_of_data_points)

    feature_index = 0
    total_features = len(feature_vectors[0])

    while feature_index < total_features:
        temp = []
        i = 0
        for feature_vector in feature_vectors:
            temp.append((i, feature_vector[feature_index], labels[i]))
            i += 1

        temp = sorted(temp, key=lambda item: item[1])

        j = 1

        left_part = list(map(lambda item: labels[item[0]], temp[:j]))
        right_part = list(map(lambda item: labels[item[0]], temp[j:]))

        left_part_counter = Counter(left_part)
        right_part_counter = Counter(right_part)

        left_part_size = len(left_part)
        right_part_size = len(right_part)

        left_part_entropy = get_entropy(left_part_counter, left_part_size)
        right_part_entropy = get_entropy(right_part_counter, right_part_size)

        temp_entropy = (((left_part_size / number_of_data_points) * left_part_entropy) + (
                (right_part_size / number_of_data_points) * right_part_entropy))

        if temp_entropy < current_entropy:
            information_gain = current_entropy - temp_entropy

            if max_information_gain < information_gain:
                splitting_feature_index = feature_index
                splitting_threshold = feature_vectors[temp[j][0]][feature_index]
                max_information_gain = information_gain
                split_tuple_index = j

        while j < number_of_data_points - 1:

            #             left_part = np.array(list(map(lambda item: labels[item[0]], temp[:j + 1])))
            #             right_part = np.array(list(map(lambda item: labels[item[0]], temp[j + 1:])))

            last_element = right_part[0]
            del right_part[0]
            left_part.append(last_element)

            left_part_counter[last_element] += 1
            right_part_counter[last_element] -= 1

            left_part_size += 1
            right_part_size -= 1

            left_part_entropy = get_entropy(left_part_counter, left_part_size)
            right_part_entropy = get_entropy(right_part_counter, right_part_size)

            temp_entropy = (((left_part_size / number_of_data_points) * left_part_entropy) + (
                    (right_part_size / number_of_data_points) * right_part_entropy))

            # if temp_entropy < current_entropy:

            information_gain = current_entropy - temp_entropy

            if max_information_gain < information_gain:
                splitting_feature_index = feature_index
                splitting_threshold = feature_vectors[temp[j][0]][feature_index]
                max_information_gain = information_gain
                split_tuple_index = j

            j += 1

        feature_index += 1

    splitting_info = None
    if splitting_feature_index is not None:
        splitting_info = {
            'splitting_feature_index': splitting_feature_index,
            'splitting_threshold': splitting_threshold,
            'max_information_gain': max_information_gain,
            'split_tuple_index': split_tuple_index
        }

    return splitting_info


# _data = get_spam_data()
# _k = 4
# _splits = k_fold_split(_k, _data)
# a=get_feature_to_split(_splits[0]['training']['labels'], _splits[0]['training']['features'])
# print(a)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def scikit_decision_tree(splits):
    testing_accuracy = []
    training_accuracy = []

    i = 0
    for split in splits:
        clf = DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(split['training']['features'], split['training']['labels'])

        predictions = clf.predict(split['training']['features'])
        training_accuracy.append(accuracy_score(split['training']['labels'], predictions))

        predictions = clf.predict(split['testing']['features'])
        testing_accuracy.append(accuracy_score(split['testing']['labels'], predictions))

        # dot_data = tree.export_graphviz(clf, out_file=None)
        # graph = graphviz.Source(dot_data)
        # graph.render("iris" + str(i))
        # i += 1

    print('Training Accuracy using Scikit Decision Tree', training_accuracy)
    print('Mean Training Accuracy using Scikit Decision Tree', np.mean(training_accuracy))

    print('Testing Accuracy using Scikit Decision Tree', testing_accuracy)
    print('Mean Testing Accuracy using Scikit Decision Tree', np.mean(testing_accuracy))


def custom_decision_tree(splits):
    training_accuracy = []
    testing_accuracy = []

    for split in splits:

        tree = create_tree(split['training']['features'], split['training']['labels'], 2, 10)

        #         print(tree)

        i = 0
        number_of_training_data_points = split['training']['labels'].size
        training_predictions = []
        while i < number_of_training_data_points:
            training_predictions.append(tree.predict_label(split['training']['features'][i]))
            i += 1

        training_accuracy.append(accuracy_score(split['training']['labels'], training_predictions))

        i = 0
        number_of_testing_data_points = split['testing']['labels'].size
        testing_error = []
        testing_predictions = []
        while i < number_of_testing_data_points:
            testing_predictions.append(tree.predict_label(split['testing']['features'][i]))
            i += 1

        testing_accuracy.append(accuracy_score(split['testing']['labels'], testing_predictions))

    print('\n')

    print('Training Accuracy using Custom Decision Tree', training_accuracy)
    print('Mean Training Accuracy using Custom Decision Tree', np.mean(training_accuracy))

    print('Testing Accuracy using Custom Decision Tree', testing_accuracy)
    print('Mean Testing Accuracy using Custom Decision Tree', np.mean(testing_accuracy))
    # print(tree)


def main():
    _data = get_spam_data()
    _data = normalize_data(_data)
    _k = 4
    _splits = k_fold_split(_k, _data, shuffle=True)
    scikit_decision_tree(_splits)
    custom_decision_tree(_splits)


#     4567

if __name__ == '__main__':
    main()
