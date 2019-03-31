from sklearn.svm import NuSVC


def demo_svm_on_spam():
    classifier = NuSVC(kernel='rbf')

if __name__ == '__main__':
    demo_svm_on_spam()