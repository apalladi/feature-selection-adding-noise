import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_evaluate_model(X_train, y_train, X_test, y_test, model):

    # scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # fit model
    print("Fitting the model with", X_train.shape[1], "features")
    model.fit(X_train, y_train)
    train_score = round(model.score(X_train, y_train), 4)
    test_score = round(model.score(X_test, y_test), 4)
    print("Train score", train_score)
    print("Test score", test_score)

    return model


def get_feature_importances(trained_model, column_names):

    # inspect coefficients
    try:
        model_coefficients = np.abs(trained_model.coef_)
    except:
        model_coefficients = trained_model.feature_importances_

    df_coef = pd.DataFrame(
        np.transpose([model_coefficients, column_names]),
        columns=["Feature importance", "Feature name"],
    )
    df_coef = df_coef.sort_values("Feature importance", ascending=False)
    df_coef.reset_index(inplace=True, drop=True)

    return df_coef


def select_relevant_features(df_coef, X, verbose):

    # select relevant features
    index_threshold = np.array(
        df_coef[df_coef["Feature name"] == "random_feature"].index
    )[0]
    relevant_features = df_coef.iloc[0:index_threshold]
    relevant_features = relevant_features["Feature name"]

    if verbose:
        print("Selected", len(relevant_features), "features out of", X.shape[1]-1)

    # return simplified dataset, containing only relevant features
    simplified_dataset = X.loc[:, relevant_features]

    return simplified_dataset


def scan_features(X, y, model, verbose):

    #  create train and test data
    X_new = X.copy(deep=True)
    X_new["random_feature"] = np.random.normal(0, 1, size=len(X_new))
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)

    trained_model = train_evaluate_model(X_train, y_train, X_test, y_test, model)

    df_coef = get_feature_importances(trained_model, X_train.columns)

    simplified_dataset = select_relevant_features(df_coef, X_new, verbose)

    return simplified_dataset


def get_relevant_features(
    X, y, model, epochs, patience, verbose=True, filename_output=False
):
    X_new = X.copy(deep=True)
    counter_patience = 0
    epoch = 0

    while (counter_patience < patience) and (epoch < epochs):
        n_features_before = X_new.shape[1]
        print("=====================EPOCH", epoch+1, "=====================")
        X_new = scan_features(X_new, y, model, verbose)
        n_features_after = X_new.shape[1]

        if n_features_before == n_features_after:
            counter_patience += 1
            print(
                "The feature selection did not improve in the last",
                counter_patience,
                "epochs",
            )
        else:
            counter_patience = 0

        epoch += 1

    if isinstance(filename_output, str):
        X_new.to_csv(filename_output, index=False)

    return X_new
