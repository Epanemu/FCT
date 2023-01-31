import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, X, y, attribute_names, dataset_name="", categorical_indicator=None):
        self.__feature_names = attribute_names
        self.__dataset_name = dataset_name
        self.__categorical_indicator = categorical_indicator

        self.__X = np.array(X, dtype=float) # the decision variable must not be a part of data, all data is already numerical
        y, self.__class_mapping = pd.factorize(y)
        self.__y = np.array(y)

        self.__n_classes = len(self.__class_mapping)
        self.__n_features = self.__X.shape[1]
        self.__generate_stats(self.__X)

    def get_training_data(self, split_seed=0, test_size=0.2, limit=np.inf, reset_stats=True):
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size=test_size, random_state=split_seed)
        X, y = self.__X_train[:limit], self.__y_train[:limit]
        self.__X_used = X
        self.__y_used = y
        if reset_stats:
            self.__generate_stats(X)
        self.__n_data = X.shape[0]
        return self.normalize(X), y

    def __generate_stats(self, X):
        X = X.copy()
        self.__shifts = X.min(axis=0)
        X -= self.__shifts
        self.__scales = X.max(axis=0)
        self.__scales[self.__scales == 0] = 1
        X /= self.__scales

        self.__epsilons = np.empty((self.n_features,))
        for i, col_data in enumerate(X.T):
            col_sorted = col_data.copy()
            col_sorted.sort()
            eps = col_sorted[1:] - col_sorted[:-1]
            eps[eps == 0] = np.inf
            self.__epsilons[i] = eps.min()

        # if all values were same (min was infinity), we want eps nonzero to prevent non-deterministic splitting
        self.__epsilons[self.__epsilons == np.inf] = 1

    def normalize(self, X):
        return (X - self.__shifts) / self.__scales

    def unnormalize(self, X):
        return X * self.__scales + self.__shifts

    @property
    def n_data(self):
        return self.__n_data

    @property
    def n_features(self):
        return self.__n_features

    @property
    def n_classes(self):
        return self.__n_classes

    @property
    def shifts(self):
        return self.__shifts

    @property
    def scales(self):
        return self.__scales

    @property
    def epsilons(self):
        return self.__epsilons

    @property
    def class_mapping(self):
        return self.__class_mapping

    @property
    def feature_names(self):
        return self.__feature_names

    @property
    def dataset_name(self):
        return self.__dataset_name

    @property
    def categorical_indicator(self):
        return self.__categorical_indicator

    @property
    def used_data(self):
        return self.__X_used, self.__y_used

    @property
    def train_data(self):
        return self.__X_train, self.__y_train

    @property
    def test_data(self):
        return self.__X_test, self.__y_test

    @property
    def all_data(self):
        return self.__X, self.__y
