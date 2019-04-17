import numpy as np

from HW_8.knn import SimilarityMeasures


class RELIEFFeatureSelection:

    @classmethod
    def select_top_features(cls, features, labels, n):
        features = features.copy()
        d = features.shape[1]
        w = np.zeros(d)

        for ix, x in enumerate(features):
            same_class_indices = labels[ix] == labels
            same_class_indices[ix] = False
            distances = SimilarityMeasures.euclidean(x, features[same_class_indices])
            z_same = features[np.argsort(distances)[0]]

            opp_class_indices = labels[ix] != labels
            opp_class_indices[ix] = False
            distances = SimilarityMeasures.euclidean(x, features[opp_class_indices])
            z_opp = features[np.argsort(distances)][0]

            w = w - np.square(x - z_same) + np.square(x - z_opp)

        return features[:, np.argsort(w)[::-1][:n]]
