"""This module produces toy dataset to be used for regression problems"""

import pandas as pd
from sklearn.datasets import make_regression


def create_regression_data():
    """This function produces a toy dataset, to be used in a regression problem.
    It takes no input and it gives as output features and labels."""

    features, labels = make_regression(
        n_samples=1000, n_features=300, n_informative=20, n_targets=1, noise=100
    )

    col_names = ["col_" + str(i) for i in range(features.shape[1])]
    features = pd.DataFrame(features, columns=col_names)
    labels = pd.DataFrame(labels, columns=["labels"])
    return features, labels
