import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_evaluate_model(x_train, y_train, x_test, y_test, model):

    # scale data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # fit model
    print("Fitting the model with", x_train.shape[1], "features")
    model.fit(x_train, y_train)
    train_score = round(model.score(x_train, y_train), 4)
    test_score = round(model.score(x_test, y_test), 4)
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


def select_relevant_features(df_coef, features, verbose):

    # select relevant features
    index_threshold = np.array(
        df_coef[df_coef["Feature name"] == "random_feature"].index
    )[0]
    relevant_features = df_coef.iloc[0:index_threshold]
    relevant_features = relevant_features["Feature name"]

    if verbose:
        print("Selected", len(relevant_features), "features out of", features.shape[1] - 1)

    # return simplified dataset, containing only relevant features
    simplified_dataset = features.loc[:, relevant_features]

    return simplified_dataset


def scan_features(features, labels, model, verbose):

    #  create train and test data
    x_new = features.copy(deep=True)
    x_new["random_feature"] = np.random.normal(0, 1, size=len(x_new))
    x_train, x_test, y_train, y_test = train_test_split(x_new, labels, test_size=0.2)

    trained_model = train_evaluate_model(x_train, y_train, x_test, y_test, model)

    df_coef = get_feature_importances(trained_model, x_train.columns)

    simplified_dataset = select_relevant_features(df_coef, x_new, verbose)

    return simplified_dataset


def get_relevant_features(
    features, labels, model, epochs, patience, verbose=True, filename_output=False
):
    x_new = features.copy(deep=True)
    counter_patience = 0
    epoch = 0

    while (counter_patience < patience) and (epoch < epochs):
        n_features_before = x_new.shape[1]
        print("=====================EPOCH", epoch + 1, "=====================")
        x_new = scan_features(x_new, labels, model, verbose)
        n_features_after = x_new.shape[1]

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
        x_new.to_csv(filename_output, index=False)

    return x_new
