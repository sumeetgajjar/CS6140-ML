import numpy as np
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.datasets import make_blobs


def demo_custom_svm_on_2d_data():
    X, y = make_blobs(n_samples=100, centers=2, n_features=2)

    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()


if __name__ == '__main__':
    np.random.seed(11)
    demo_custom_svm_on_2d_data()
