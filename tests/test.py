import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from src.data import create_regression_data
from src.ml import (
    train_evaluate_model,
    get_feature_importances,
    compute_mean_coefficients,
    select_relevant_features,
    generate_kfold_data,
)


def test_data_creation():
    features, labels = create_regression_data()
    assert features.shape == (1000, 300), "Shape of features is wrong"
    assert labels.shape == (1000, 1), "Shape of labels is wrong"


def test_train_evaluate():
    x_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)
    x_test = np.random.rand(20, 10)
    y_test = np.random.rand(20, 1)
    model = Lasso()
    trained_model = train_evaluate_model(
        x_train, y_train, x_test, y_test, model, verbose=False
    )
    assert (
        len(trained_model.coef_) == x_train.shape[1]
    ), "The model is not trained properly"


def test_get_features_importance():
    x_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    column_names = np.arange(0, x_train.shape[1])

    lasso = Lasso()
    lasso.fit(x_train, y_train)
    df_coef = get_feature_importances(lasso, column_names)
    assert (
        type(df_coef) == pd.DataFrame
    ), "The table with feature importance must be a DataFrame"
    assert (
        len(df_coef) == x_train.shape[1]
    ), "The number of coefficients does not match the shape of the training data"

    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)
    df_coef = get_feature_importances(rf, column_names)
    assert (
        type(df_coef) == pd.DataFrame
    ), "The table with feature importance must be a DataFrame"
    assert (
        len(df_coef) == x_train.shape[1]
    ), "The number of coefficients does not match the shape of the training data"


def test_mean_coefficients_single_column():
    feature_importance = np.random.randint(-100, 100, size=20)
    df = pd.DataFrame(feature_importance, index=np.arange(0, len(feature_importance)))
    vec = np.sort(np.abs(feature_importance))[::-1]
    df_sorted = compute_mean_coefficients(df)
    assert all(
        np.array(df_sorted["Feature importance"]) == vec
    ), "Feature importances are not sorted properly"


def test_mean_coefficients_multiple_columns():
    feature_importance = 2 * np.random.rand(100, 5) - 1
    df = pd.DataFrame(feature_importance, index=np.arange(0, len(feature_importance)))
    vec = np.sort(np.abs(df.mean(axis=1)))[::-1]
    df_sorted = compute_mean_coefficients(df)
    assert all(
        np.array(df_sorted["Feature importance"]) == vec
    ), "Feature importances are not sorted properly"


def test_select_relevant_features():
    df_coef = pd.DataFrame([5, 4, 3, 2, 1], columns=["Feature importance"])
    df_coef["Feature name"] = ["col1", "col2", "random_feature", "col3", "col4"]
    features = pd.DataFrame(
        np.random.rand(10, 5),
        columns=["col1", "col2", "random_feature", "col3", "col4"],
    )
    feature_selected = select_relevant_features(df_coef, features, verbose=True)
    assert all(feature_selected.columns == ["col1", "col2"]), "Wrong columns selected"


def test_kfold_splitting():
    features = pd.DataFrame(np.random.rand(100, 10))
    labels = pd.DataFrame(np.random.rand(100))
    x_trains, y_trains, x_tests, y_tests = generate_kfold_data(
        features, labels, random_state=42
    )
    assert len(x_trains) == 5, "Length of train features is wrong"
    assert len(x_tests) == 5, "Length of test features is wrong"
    assert len(y_trains) == 5, "Length of train labels is wrong"
    assert len(y_tests) == 5, "Length of test labels is wrong"
