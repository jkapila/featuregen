import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import VarianceThreshold


class CartesianProduct(BaseEstimator, TransformerMixin):
    """

    Creating a Cartesian product for data

    """

    def __init__(self):
        self.multiplier_df = None
        self.multiplier_dtypes = None
        self.flag = None

    def _cartesian_product_simplified(self, left, right, dtypes):
        la = len(left)
        lb = len(right)
        ia2, ib2 = np.broadcast_arrays(*np.ogrid[:la, :lb])
        columns = left.columns.tolist() + right.columns.tolist()
        df = pd.DataFrame(np.column_stack([left.values[ia2.ravel()],
                        right.values[ib2.ravel()]]), columns=columns)
        df = df.astype(dtypes)
        return df

    def fit(self, X, y=None):

        """
        :param X: Pandas data frame to be used for cartesion
        """

        # do nothing
        self.multiplier_df = X
        self.multiplier_dtypes = X.dtypes

        return self

    def transform(self, X, y=None, flag="predict"):
        """
        Transforming

        :param X: Pandas DataFrame
        """

        self.flag = flag
        data = self.multiplier_df.copy(deep=True)

        if self.flag == "predict":
            df = self._cartesian_product_simplified(X, data, X.dtypes.append(data.dtypes))
            print("Added Features : ", self.multiplier_df.columns.tolist())
            print("Post Transforming with Cartesian Multiplied Features data shape is:", df.shape)
        else:
            df = X
        return df


class TrainingFeatures(BaseEstimator, TransformerMixin):

    """

    Finalizing Features from data

    :param target: Target attribute
    :param date_col: Date Column
    :param drop_features: Features to drop from Training

    """

    def __init__(self, target=None, date_col="date", drop_features=None):
        self.features = None
        self.target = target
        self.date_col = date_col
        self.drop_features = drop_features if isinstance(drop_features, list) else [drop_features]

    def fit(self, X):
        """
        Fitting

        :param X: Pandas DataFrame
        """

        # Remove features with 0 variance
        X = X[X.columns[~X.columns.isin(self.drop_features)]]
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        constant_filter = VarianceThreshold(threshold=0)
        constant_filter.fit(X)
        constant_columns = list(X.columns[constant_filter.variances_ == 0])

        print("Constant Features: " + str(constant_columns))

        if len(constant_columns) > 1:
            X = X.drop(columns=constant_columns, axis=1)

        # Remove duplicated columns (columns that have exact same values)
        X_T = X.T
        unique_features = X_T.drop_duplicates(keep="first").T
        duplicated_features = [dup_col for dup_col in X.columns if dup_col not in unique_features.columns]

        print("Duplicated Features: " + str(duplicated_features))

        if len(duplicated_features) > 1:
            X = X.drop(columns=duplicated_features, axis=1)

        self.features = X.columns.tolist()
        return self

    def transform(self, X, with_target=False):
        """
        Transforming

        :param X: Pandas DataFrame
        """
        if with_target:
            df = X[self.features].copy()
            df["target"] = X[self.target]
            df[self.date_col] = X[self.date_col]
            return df

        return X[self.features]
