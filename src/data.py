import pandas as pd
from sklearn.datasets import make_regression


def create_regression_data():
    X, y = make_regression(
        n_samples=1000, n_features=300, n_informative=20, n_targets=1, noise=100
    )

    col_names = ["col_" + str(i) for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.DataFrame(y, columns=['labels'])
    return X, y
