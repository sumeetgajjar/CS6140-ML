from scipy import sparse
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import BinaryRelevance


def get_instance_f1(pred, ground_file):
    c = 0
    count = 0
    sum_f1 = 0
    sum_acc = 0
    for l1, l2 in zip(pred, (ground_file)):
        l2 = [int(i) for i in l2]
        if l1 == l2:
            sum_acc += 1
        c += 1
        count += 1
        overlap = 0
        for i in l1:
            if i in l2:
                overlap += 1

        if (len(l1) + len(l2)) == 0:
            sum_f1 += 1
        else:
            sum_f1 += (2 * overlap) / (len(l1) + len(l2))

    return sum_f1 / count, sum_acc / count


tran = MultiLabelBinarizer()

x_train, y_train = load_svmlight_file("all_train.csv", multilabel=True, n_features=30, zero_based=True)
x_test, y_test = load_svmlight_file("all_test.csv", multilabel=True, n_features=30, zero_based=True)

y_train2 = tran.fit_transform(y_train)
y_train2 = sparse.csr_matrix(y_train2)

y_test2 = tran.fit_transform(y_test)
y_test2 = sparse.csr_matrix(y_test2)

clf = BinaryRelevance(
    classifier=LogisticRegression(solver='liblinear', penalty="l1", C=0.1,
                                  multi_class='ovr',
                                  tol=1e-8, max_iter=100),
    require_dense=[False, True])
clf.fit(x_train, y_train2)
predsTrn = ([list(line.nonzero()[1]) for line in clf.predict(x_train)])
tr_f1, tr_acc = get_instance_f1(predsTrn, y_train)
print("datasets\taccuracy\tf1")
print("train\t" + str(tr_f1) + "\t" + str(tr_acc))

preds = ([list(line.nonzero()[1]) for line in clf.predict(x_test)])
tst_f1, tst_acc = get_instance_f1(preds, y_test)
print("test\t" + str(tst_f1) + "\t" + str(tst_acc))

# print the weights for each logistic regression
for i in range(10):
    lr = clf.classifiers_[i]
    coefs = lr.coef_
    print("label " + str(i) + " coefs: " + str(coefs))
