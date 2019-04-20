import numpy as np
from scipy import sparse
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import LabelPowerset

from FINALS.pb3.trainBR import get_instance_f1


class GroupClassifier:

    def __init__(self, label_groups, no_of_labels) -> None:
        super().__init__()
        self.no_of_labels = no_of_labels
        self.label_groups = label_groups
        self.classifiers = []

    def fit(self, features, labels):

        for label_group in self.label_groups:

            temp_labels = np.zeros((features.shape[0], self.no_of_labels), dtype=int)
            for ix, x in enumerate(features):
                for _label in label_group:
                    temp_labels[ix, _label] = labels[ix, _label]

            _classifier = LabelPowerset(LogisticRegression(solver='liblinear', penalty="l1", C=0.1,
                                                           multi_class='ovr',
                                                           tol=1e-8, max_iter=100))

            _classifier.fit(features, sparse.csr_matrix(temp_labels))
            self.classifiers.append(_classifier)

    def predict(self, features):

        predictions = np.zeros((features.shape[0], self.no_of_labels))

        for _classifier in self.classifiers:
            temp_pred = ([list(line.nonzero()[1]) for line in _classifier.predict(features)])
            for ix, _pred in enumerate(temp_pred):
                if _pred:
                    predictions[ix, _pred] = 1

        return predictions


def create_groups():
    cov = EmpiricalCovariance().fit(y_train2)
    a = np.array(cov.covariance_)

    print(a)
    for i in range(10):

        temp = []
        for j in range(10):

            if i != j and j > i and a[i][j] >= 0.003:
                temp.append(j)

        print(i, " => ", temp)


tran = MultiLabelBinarizer()
x_train, y_train = load_svmlight_file("all_train.csv", multilabel=True, n_features=30, zero_based=True)
x_test, y_test = load_svmlight_file("all_test.csv", multilabel=True, n_features=30, zero_based=True)

y_train2 = tran.fit_transform(y_train)

y_test2 = tran.fit_transform(y_test)

groups = [[0, 1, 2], [3, 4, 5, 9], [6, 7, 8]]
group_classifier = GroupClassifier(groups, 10)
group_classifier.fit(x_train, y_train2)

y_train_pred = group_classifier.predict(x_train)
training_f1, training_acc = get_instance_f1(y_train_pred, y_train)
print("train\t" + str(training_f1) + "\t" + str(training_acc))

y_test_pred = group_classifier.predict(x_test)
testing_f1, testing_acc = get_instance_f1(y_test_pred, y_test)
print("test\t" + str(testing_f1) + "\t" + str(testing_acc))
