from sklearn.metrics import accuracy_score

from HW_8 import utils
from HW_8.knn import KNN, Kernel


def demo_knn_on_spam_base_data():
    print("+" * 40, "KNN on Spambase Data with Euclidean Kernel", "+" * 40)
    data = utils.get_spam_data()
    data = utils.k_fold_split(10, data, 11, shuffle=True)[0]

    training_features = data['training']['features']
    training_labels = data['training']['labels']
    testing_features = data['testing']['features']
    testing_labels = data['testing']['labels']

    for k in [1, 3, 7]:
        classifier = KNN(Kernel.euclidean, training_features, training_labels, verbose=0)

        pred_training_labels = classifier.predict(training_features, k)
        acc = accuracy_score(training_labels, pred_training_labels)
        print("k=>{}, Training Accuracy:{}".format(k, acc))

        pred_testing_labels = classifier.predict(testing_features, k)
        acc = accuracy_score(testing_labels, pred_testing_labels)
        print("k=>{}, Testing Accuracy:{}".format(k, acc))
        print()

    print("+" * 40, "KNN on Spambase Data with Euclidean Kernel", "+" * 40)


if __name__ == '__main__':
    demo_knn_on_spam_base_data()
