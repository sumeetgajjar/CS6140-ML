import numpy as np
import pandas as pd


class Node(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.splitting_info = None
        self.price = -1

    def add_left(self, node):
        self.left = node

    def add_right(self, node):
        self.right = node

    def __str__(self, level=0):
        ret = "{} Feature={}, Threshold={}, Price={} \n\n".format(
            "\t" * level, self.splitting_info['splitting_feature_index'], self.splitting_info['splitting_threshold'],
            self.price)

        if self.left is not None:
            ret += self.left.__str__(level + 1)
        if self.right is not None:
            ret += self.right.__str__(level + 1)

        return ret

    def predict_price(self, feature_vector):
        if self.left is None or self.right is None:
            return self.price

        splitting_threshold = self.splitting_info['splitting_threshold']
        splitting_feature_index = self.splitting_info['splitting_feature_index']
        if feature_vector[splitting_feature_index] <= splitting_threshold:
            return self.left.predict_price(feature_vector)
        else:
            return self.right.predict_price(feature_vector)


def normalize_data(data):
    feature_vectors = data['features'];

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


def get_housing_data():
    training_data = pd.read_csv('data/housing/housing_train.txt', delimiter='\\s+', header=None)
    training_features = np.array(training_data.iloc[:, 0:13])
    training_labels = np.array(training_data.iloc[:, 13])

    testing_data = pd.read_csv('data/housing/housing_test.txt', delimiter='\\s+', header=None)
    testing_features = np.array(testing_data.iloc[:, 0:13])
    testing_labels = np.array(testing_data.iloc[:, 13])

    if training_features.shape[0] != training_labels.shape[0]:
        raise Exception("Mismatch in Training Feature Tuples(%s) and Label Tuples(%s)" % (
            training_features.shape, training_labels.shape))

    if testing_features.shape[0] != testing_labels.shape[0]:
        raise Exception("Mismatch in Testing Feature Tuples(%s) and Label Tuples(%s)" % (
            testing_features.shape, testing_labels.shape))

    return {
        'training': {
            'features': training_features,
            'prices': training_labels
        },
        'testing': {
            'features': testing_features,
            'prices': testing_labels
        }
    }


def split_data(splitting_info, prices, feature_vectors):
    splitting_threshold = splitting_info['splitting_threshold']
    splitting_feature_index = splitting_info['splitting_feature_index']

    left_part_feature_vectors = []
    left_part_prices = []
    right_part_feature_vectors = []
    right_part_prices = []

    number_of_data_points = len(feature_vectors)
    i = 0
    while i < number_of_data_points:
        if feature_vectors[i][splitting_feature_index] <= splitting_threshold:
            left_part_feature_vectors.append(feature_vectors[i])
            left_part_prices.append(prices[i])
        else:
            right_part_feature_vectors.append(feature_vectors[i])
            right_part_prices.append(prices[i])

        i += 1

    return {
        'left': {
            'features': np.array(left_part_feature_vectors),
            'prices': np.array(left_part_prices)
        },
        'right': {
            'features': np.array(right_part_feature_vectors),
            'prices': np.array(right_part_prices)
        }
    }


def get_feature_to_split(prices, feature_vectors):
    current_mean = prices.mean()
    current_mse = np.square(prices - current_mean).mean()

    min_mse = current_mse
    splitting_feature_index = -1
    splitting_threshold = -1

    number_of_data_points = len(prices)

    feature_index = 0
    total_features = len(feature_vectors[0])
    while feature_index < total_features:
        temp = []
        i = 0
        for feature_vector in feature_vectors:
            temp.append((i, feature_vector[feature_index]))
            i += 1

        temp = sorted(temp, key=lambda item: item[1])

        j = 0
        while j < number_of_data_points - 1:

            left_part = np.array(list(map(lambda item: prices[item[0]], temp[:j + 1])))
            right_part = np.array(list(map(lambda item: prices[item[0]], temp[j + 1:])))

            left_part_mean = left_part.mean()
            right_part_mean = right_part.mean()

            temp_mse = np.append(np.square(left_part - left_part_mean), np.square(right_part - right_part_mean)).mean()

            if min_mse > temp_mse:
                splitting_feature_index = feature_index
                splitting_threshold = feature_vectors[temp[j][0]][feature_index]
                min_mse = temp_mse

            j += 1

        feature_index += 1

    return {
        'splitting_feature_index': splitting_feature_index,
        'splitting_threshold': splitting_threshold,
        'min_mse': min_mse
    }


# data = get_housing_data()
# prices = data['training']['labels']
# features_vectors = data['training']['features']
# get_feature_to_split(prices, features_vectors)


def create_tree_util(current_level, feature_vectors, prices, min_sample_size, max_depth):
    if prices.size <= min_sample_size or current_level > max_depth:
        return None

    splitting_info = get_feature_to_split(prices, feature_vectors)
    splitted_data = split_data(splitting_info, prices, feature_vectors)
    if splitted_data['left']['prices'].size == 0 or splitted_data['right']['prices'].size == 0:
        return None

    root = Node()
    root.price = prices.mean()
    root.splitting_info = splitting_info
    root.left = create_tree_util(current_level + 1, splitted_data['left']['features'], splitted_data['left']['prices'],
                                 min_sample_size, max_depth)
    root.right = create_tree_util(current_level + 1, splitted_data['right']['features'],
                                  splitted_data['right']['prices'], min_sample_size, max_depth)

    return root


def create_tree(feature_vectors, prices, min_sample_size, max_depth):
    return create_tree_util(1, feature_vectors, prices, min_sample_size, max_depth)


from sklearn.tree import DecisionTreeRegressor


def custom_decision_tree(data):
    tree = create_tree(data['training']['features'], data['training']['prices'], 2, 10)

    i = 0
    number_of_training_data_points = data['training']['prices'].size
    training_error = []
    while i < number_of_training_data_points:
        training_error.append(tree.predict_price(data['training']['features'][i]) - data['training']['prices'][i])
        i += 1

    training_error = np.array(training_error)
    print('Training MSE using Custom Decision Tree', np.square(training_error).mean())

    i = 0
    number_of_testing_data_points = data['testing']['prices'].size
    testing_error = []
    while i < number_of_testing_data_points:
        testing_error.append(tree.predict_price(data['testing']['features'][i]) - data['testing']['prices'][i])
        i += 1

    testing_error = np.array(testing_error)
    print('Testing MSE using Custom Decision Tree', np.square(testing_error).mean())


#     print(tree)

def scikit_decision_tree(data):
    s_tree = DecisionTreeRegressor(min_samples_leaf=2, max_depth=10)
    s_tree.fit(data['training']['features'], data['training']['prices'])

    predicted_prices = s_tree.predict(data['training']['features'])
    training_error = predicted_prices - data['training']['prices']
    print('Training MSE using Scikit Decision Tree', np.square(training_error).mean())

    predicted_prices = s_tree.predict(data['testing']['features'])
    testing_error = predicted_prices - data['testing']['prices']
    print('Testing MSE using Scikit Decision Tree', np.square(testing_error).mean())


def main():
    data = get_housing_data()
    #     data = normalize_data(data)
    custom_decision_tree(data)
    scikit_decision_tree(data)


#     1324

if __name__ == '__main__':
    main()
