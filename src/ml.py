"""This module contains the function to perform the feature selection,
by adding random noise"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.base import BaseEstimator

from typing import Union,Tuple,List,Optional

def train_evaluate_model(x_train:pd.DataFrame, 
                         y_train:pd.DataFrame, 
                         x_test:pd.DataFrame,
                         y_test:pd.DataFrame, 
                         model:BaseEstimator, 
                         verbose:bool) -> BaseEstimator:

    print("vediamo")
    """It trains and evaluate the machine learning model.

    Parameters:
        - x_train: training features
        - y_train: training labels
        - x_test: test features
        - y_test: test labels
        - model: a scikit-learn machine learning (untrained) model

    Return:
        - the trained model
    """

    # scale data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # fit model
    if verbose:
        print("Fitting the model with", x_train.shape[1], "features")
    model.fit(x_train, y_train)
    train_score = round(model.score(x_train, y_train), 4)
    test_score = round(model.score(x_test, y_test), 4)

    if verbose:
        print("Train score", train_score)
        print("Test score", test_score)

    return model


def get_feature_importances(trained_model:Union[Lasso,RandomForestClassifier], 
                            column_names:List[str]) -> pd.DataFrame:
    """It computes the features importance, given a trained model.

    Parameters:
        - trained_model: a scikit-learn ML trained model
        - column_names: the name of the columns associated to the features

    Return:
        - a DataFrame containing the feature importance (not sorted) as column and
        the name of the features as index
    """

    # inspect coefficients
    if hasattr(trained_model, 'coef_'):
        model_coefficients = trained_model.coef_
    elif hasattr(trained_model, 'feature_importances_'):
        model_coefficients = trained_model.feature_importances_
    else:
        raise ValueError("Could not retrieve the feature importance")

    df_coef = pd.DataFrame(model_coefficients, index=column_names)

    return df_coef


def compute_mean_coefficients(df_coefs:pd.DataFrame) -> pd.DataFrame:
    """It computes the average coefficients, given a DataFrame with multiple columns.

    Parameters:
        - a DataFrame with coefficients obtained in multiple trainings

    Return:
        - a DataFrame with one column, containing the absolute values of the average coefficients
    """

    if df_coefs.shape[1] > 1:
        df_coef = pd.DataFrame(df_coefs.mean(axis=1), columns=["Feature importance"])
    else:
        print("Using this one")
        df_coef = pd.DataFrame(df_coefs.iloc[:, 0], index=df_coefs.index)
        df_coef.columns = ["Feature importance"]

    df_coef["Feature importance"] = np.abs(df_coef["Feature importance"])
    df_coef["Feature name"] = df_coef.index
    df_coef = df_coef.sort_values("Feature importance", ascending=False)
    df_coef.reset_index(inplace=True, drop=True)

    return df_coef


def select_relevant_features(df_coef:pd.DataFrame, 
                             features:pd.DataFrame, 
                             verbose:bool) -> pd.DataFrame:
    """It computes the relevant features, given the DataFrame with feature importance
    and the original features.
    This is obtained by adding a feature with random noise.

    Parameters:
        - df_coef: the DataFrame with the the feature importance
        - features: the original features
        - verbose: True or False to tune the level of verbosity

    Return:
        - the simplified dataset, with the relevant features
    """

    # select relevant features
    index_threshold = np.array(
        df_coef[df_coef["Feature name"] == "random_feature"].index
    )[0]
    relevant_features = df_coef.iloc[0:index_threshold]
    relevant_features = relevant_features["Feature name"]

    if verbose:
        print(
            "Selected", len(relevant_features), "features out of", features.shape[1] - 1
        )

    # return simplified dataset, containing only relevant features
    simplified_dataset = features.loc[:, relevant_features]

    return simplified_dataset


def generate_kfold_data(features:pd.DataFrame, labels:pd.DataFrame, random_state:int
                        ) -> Tuple[List,List,List,List]:
    """It splits the data into training and validation,
    by using the KFold splitting method.

    Parameters:
        - features: the matrix with features, commonly called X
        - labels: the vector with labels, commonly called y

    Return:
        - train and test data
    """

    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []

    k_fold = KFold(n_splits=5, random_state=random_state, shuffle=True)
    k_fold.get_n_splits(features)
    for _, (train_index, test_index) in enumerate(k_fold.split(features)):
        # train data
        x_trains.append(features.iloc[train_index, :])
        y_trains.append(labels.iloc[train_index])
        # test data
        x_tests.append(features.iloc[test_index, :])
        y_tests.append(labels.iloc[test_index])

    return x_trains, y_trains, x_tests, y_tests


def train_with_kfold_splitting(features:pd.DataFrame, labels:pd.DataFrame,
                                model:BaseEstimator, 
                               verbose:bool, random_state:int) -> pd.DataFrame:
    """It trains the model using the kfold splitting and returns
    a DataFrame with the feature importance.

    Parameters:
        - features: the matrix with features, commonly called X
        - labels: the vector with labels, commonly called y
        - model: an untrained scikit-learn model
        - verbose: True or False to tune the level of verbosity
        - random_state: select the random state of the train/test splitting process

    Return:
        - a DataFrame with one column, containing the features importance (or the coefficients)
    """

    # create train-test data
    x_trains, y_trains, x_tests, y_tests = generate_kfold_data(
        features, labels, random_state
    )

    for i in range(len(x_trains)):
        trained_model = train_evaluate_model(
            x_trains[i], y_trains[i], x_tests[i], y_tests[i], model, verbose
        )
        if i == 0:
            df_coefs = get_feature_importances(trained_model, x_trains[i].columns)
            df_coefs.columns = ["cycle_" + str(i + 1)]
        else:
            df_coefs["cycle_" + str(i + 1)] = get_feature_importances(
                trained_model, x_trains[i].columns
            )

    df_coef = compute_mean_coefficients(df_coefs)
    return df_coef


def train_with_simple_splitting(features:pd.DataFrame, labels:pd.DataFrame,
                                model:BaseEstimator, verbose:bool, 
                                random_state:int) -> pd.DataFrame:
    """It trains the model using the train/test splitting and returns
    a DataFrame with the feature importance.

    Parameters:
        - features: the matrix with features, commonly called X
        - labels: the vector with labels, commonly called y
        - model: an untrained scikit-learn model
        - verbose: True or False to tune the level of verbosity
        - random_state: select the random state of the train/test splitting process

    Return:
        - a DataFrame with one column, containing the features importance (or the coefficients)
    """

    # create train-test data
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=random_state
    )

    trained_model = train_evaluate_model(
        x_train, y_train, x_test, y_test, model, verbose
    )
    df_coefs = get_feature_importances(trained_model, x_train.columns)

    df_coef = compute_mean_coefficients(df_coefs)

    return df_coef


def scan_features_pipeline(features:pd.DataFrame, labels:pd.DataFrame,
                            model:BaseEstimator, splitting_type:str, 
                            verbose:bool, random_state:int) -> pd.DataFrame:
    """This pipeline performs various operations:
    - train and evaluate the model
    - generates the DataFrame with the feature importance
    - computes the simplified dataset, containing only the relevant features

    Parameters:
        - features: the matrix with features, commonly called X
        - labels: the vector with labels, commonly called y
        - model: an untrained scikit-learn model
        - splitting_type: choose between "simple" (80% train, 20% test)
          or "kfold" (5-fold splitting)
        - verbose: True or False to tune the level of verbosity
        - random_state: select the random state of the train/test splitting process

    Return:
        - the simplified dataset, containing only the most relevant features
    """

    #  add noise
    x_new = features.copy(deep=True)
    x_new["random_feature"] = np.random.normal(0, 1, size=len(x_new))

    if splitting_type == "kfold":
        df_coef = train_with_kfold_splitting(
            x_new, labels, model, verbose, random_state
        )
    elif splitting_type == "simple":
        df_coef = train_with_simple_splitting(
            x_new, labels, model, verbose, random_state
        )
    else:
        raise ValueError("Choice not recognized. Possible choices are kfold or simple")

    simplified_dataset = select_relevant_features(df_coef, x_new, verbose)

    return simplified_dataset


def get_relevant_features(
    features:pd.DataFrame,
    labels:pd.DataFrame,
    model:BaseEstimator,
    splitting_type:str,
    epochs:int,
    patience:int,
    verbose:bool=True,
    filename_output:Optional[str]=None,
    random_state:int=42,
) -> pd.DataFrame:
    """This functions performs multiple cycles to reduce the dimension of the dataset.

    Parameters:
        - features: the matrix with features, commonly called X
        - labels: the vector with labels, commonly called y
        - model: an untrained scikit-learn model
        - splitting_type: choose between "simple" (80% train, 20% test)
          or "kfold" (5-fold splitting)
        - epochs: the number of epochs (or cycles)
        - patience: the number of cycles of non-improvement to wait before stopping
        the execution of the code
        - verbose: True or False, to tune the level of verbosity
        - filename_output:  name of the simplified dataset if you want to export it, default is None
        - random_state: select the random seed

    Return:
        - the dataset simplified after multiple epochs of feature selection
    """

    x_new = features.copy(deep=True)
    counter_patience = 0
    epoch = 0

    np.random.seed(random_state)
    random_states = np.random.randint(1, int(10 * epochs), size=epochs)

    while (counter_patience < patience) and (epoch < epochs):
        n_features_before = x_new.shape[1]
        print("=====================EPOCH", epoch + 1, "=====================")
        x_new = scan_features_pipeline(
            x_new, labels, model, splitting_type, verbose, random_states[epoch]
        )
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

    if filename_output is not None:
        x_new.to_csv(filename_output, index=False)

    return x_new
