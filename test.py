from src.data import create_regression_data

def test_data_creation():
    features, labels = create_regression_data()
    assert features.shape == (1000, 300), "Shape of features is wrong"
    assert labels.shape == (1000, 1), "Shape of labels is wrong"