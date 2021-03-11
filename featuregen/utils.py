"""
Utility functions and classes
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def check_value_in_array(value, array):
    return value in array


def check_values_in_array(values, array):
    cond = []
    for val in values:
        cond.append(check_value_in_array(val, array))
    return any(cond)


class XFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y):
        pass

    def transform(self, X):
        pass

    def predict(self, X):
        pass
