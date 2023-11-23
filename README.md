# feature-selection-adding-noise

## Introduction
The purpose of this small library is to apply feature selection to your high-dimensional data. In order to do that, we apply the following steps:
1) the input features are automatically standardized
2) a column containing gaussian noise (average = 0, standard deviation = 1) is added
3) a model is trained
4) the feature importance is evaluated
5) all the features that are less important than the random one, are excluded
The previous steps are repeated iteratively, until the algorithm converges.

## Initialize the repository
Let us start by cloning the repository, by using the following command:
```
git@github.com:apalladi/feature-selection-adding-noise.git
```
Then you need to install the dependencies. I suggest to create a virtual environment, as follows:
```
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

To check if everything works, you can run the unit tests:
```
python -m pytest tests/test.py
```

You are now ready to use the repository!

## Getting started
The example you need is contained in [this notebook](example.ipynb).
A toy dataset, to build a regression model, is imported. 
Then we import the function `get_relevant_features`
```
from src.ml import get_relevant_features
```
This function takes as arguments:
- `features`
- `labels`
- `model`, a scikit-learn model
- `epochs`, the number of epochs (i.e. for how many cycles you want to apply recursively the feature selection)
- `patience`, number of epochs without any improvement of the features selection, before stopping the process (the idea is similar to the early stopping of Tensorflow/Keras)
- `splitting_type`, it can be equal to `simple` (for simple train/test split) or `kfold` (for 5-fold splitting). If you choose `kfold`, the feature importance will be computed as the average feature importance for each train/test subset.
- `noise_type`, it can be equal to `gaussian` for gaussian noise or `random` for flat random noise
- `importance_type`, it can be equal to `model` for using model coefficients or `shap` for extracting importance using Shapley values
- `filename_output`, a string to indicate where to save the file. You can also choose `None` if you do not want to save it
- `random_state`, set the random seed that it is used by the k-fold splitting

The function `get_relevant_features` returns a DataFrame with a reduced dataset, i.e. a dataset that contains only the most important features.

