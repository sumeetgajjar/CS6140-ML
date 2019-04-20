import numpy as np
import pandas as pd

np.random.seed(11)


class BernoulliEM:

    def __init__(self, t, k) -> None:
        super().__init__()
        self.t = t
        self.k = k
        self.gamma = None
        self.mixture_coefficients = None
        self.q = None

    def e_step(self, data):
        no_of_heads = np.sum(data, axis=1)
        heads_prob = np.power(np.repeat([self.q], data.shape[0], axis=0),
                              np.reshape(no_of_heads, (no_of_heads.shape[0], 1)))

        no_of_tails = self.k - no_of_heads
        tails_prob = np.power(np.repeat([1 - self.q], data.shape[0], axis=0),
                              np.reshape(no_of_tails, (no_of_tails.shape[0], 1)))

        num = (self.mixture_coefficients * heads_prob * tails_prob).T
        self.gamma = num / num.T.sum(axis=1)

    def m_step(self, data):
        self.mixture_coefficients = np.sum(self.gamma, axis=1) / data.shape[0]

        y = np.sum(data, axis=1)
        self.q = np.sum(self.gamma * y, axis=1) / (np.sum(self.gamma, axis=1) * self.k)

    def train(self, data, epoch):
        self.mixture_coefficients = np.random.random(self.t)
        self.mixture_coefficients = self.mixture_coefficients / np.sum(self.mixture_coefficients)

        self.q = np.random.uniform(0, 1, self.t)

        for i in range(epoch):
            self.e_step(data)
            self.m_step(data)

    @staticmethod
    def generate_data(t, m, k, coin_selection_prob, coin_head_prob):
        selected_coins = np.random.choice(t, m, p=coin_selection_prob)
        sequences = []
        for coin in selected_coins:
            sequences.append(np.random.binomial(1, coin_head_prob[coin], k))

        return np.array(sequences)


def demo_bernoulli_em():
    data = pd.read_csv('OUT.txt', delimiter='\\s+', header=None)
    data = np.array(data.iloc[:, :])

    t = 3
    k = 20
    em = BernoulliEM(t, k)
    em.train(data, 100)
    print("Problem assignment Probability", em.mixture_coefficients)
    print("Probability of solving each problem", em.q)


if __name__ == '__main__':
    demo_bernoulli_em()
