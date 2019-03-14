import numpy as np

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

    def train(self, data, epoch, display_step):
        # self.gamma = np.random.random((self.t, data.shape[0]))
        # self.gamma = self.gamma / np.sum(self.gamma, axis=0)

        self.mixture_coefficients = np.random.random(self.t)
        self.mixture_coefficients = self.mixture_coefficients / np.sum(self.mixture_coefficients)

        self.q = np.random.uniform(0, 1, self.t)

        for i in range(epoch):
            self.e_step(data)
            self.m_step(data)

            # if i % display_step == 0 or i == 0:
            #     self.plot(data)

    def plot(self, data):
        print("q", self.q)
        print("pie", self.mixture_coefficients)

    @staticmethod
    def generate_data(t, m, k, coin_selection_prob, coin_head_prob):
        selected_coins = np.random.choice(t, m, p=coin_selection_prob)
        sequences = []
        for coin in selected_coins:
            sequences.append(np.random.binomial(1, coin_head_prob[coin], k))

        return np.array(sequences)


def demo_em_on_coins():
    t = 2
    m = 1000
    k = 10
    em = BernoulliEM(t, k)
    data = em.generate_data(t, m, k, coin_selection_prob=[0.8, 0.2], coin_head_prob=[0.75, 0.4])
    em.train(data, 100, 1)
    print("Coin selection Probability", em.mixture_coefficients)
    print("Heads Probability", em.q)


if __name__ == '__main__':
    demo_em_on_coins()
