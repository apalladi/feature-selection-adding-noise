import numpy as np
from sklearn.linear_model import Lasso
from src.data import create_regression_data
from src.ml import train_evaluate_model

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
    trained_model = train_evaluate_model(x_train, y_train, x_test, y_test, model, verbose=False)
    assert len(trained_model.coef_) == x_train.shape[1], "The model is not trained properly"
    