import numpy as np


class ECOC:

    def __init__(self) -> None:
        super().__init__()

    def train(self):
        pass

    def predict(self):
        pass

    @staticmethod
    def generate_ecoc_exhaustive_code(no_of_classes):
        no_of_bits = 2 ** (no_of_classes - 1)

        codes = [[1 for _ in range(no_of_bits)]]
        for i in range(2, no_of_classes + 1):
            no_of_ones = 2 ** (no_of_classes - i)
            no_of_zeros = 2 ** (no_of_classes - i)

            code = []
            for j in range(2 ** (i - 2)):
                code.extend(np.zeros(no_of_zeros).tolist())
                code.extend(np.ones(no_of_ones).tolist())

            codes.append(code)

        codes = np.array(codes)
        codes = codes[:, :-1].copy()
        return codes


def demo_ecoc_on_8_news_group_data():
    codes = ECOC.generate_ecoc_exhaustive_code(5)
    print(codes.shape)


if __name__ == '__main__':
    demo_ecoc_on_8_news_group_data()
