import numpy as np
import pandas as pd


## Grouped Variable Transformation to transform data in various groups
class GroupedVariableTransformation:
    """

    Variable Transformation at group level

    :param key: Grouping / ID column against which aggregatoin would happen
    :param target: Target column name which needs to be used for creating aggregation
    :param monotone_constraints: Montonic Constraints


    :param strategy: 'zscore','min-max','median-iqr', mean-iqr','median-qd',
                    'median-std','median-range','mean-range','min-qd','min-iqr',
                    'min','max','mean','median','std','iqr','qd','range','custom'
    :param custom_func: The function should take 4 arguments
                        custom_func(value,group,groups_estimates,inverse)

    """

    def __init__(self, key, target, strategy="zscore", custom_func=None):

        self.key = key
        self.target = target
        self.strategy = strategy
        self.custom_func = custom_func

        def summary_func(x):
            # First quartile (Q1)
            Q1 = np.nanpercentile(x, 25, interpolation="midpoint")

            # Third quartile (Q3)
            Q3 = np.nanpercentile(x, 75, interpolation="midpoint")

            op = dict(
                min_val=np.nanmin(x),
                max_val=np.nanmax(x),
                mean_val=np.nanmean(x),
                median_val=np.nanmedian(x),
                std_val=np.nanstd(x),
                range_val=np.nanmax(x) - np.nanmin(x),
                iqr_val=Q3 - Q1,
                qd_val=(Q3 - Q1) / 2,
            )
            return op

        self.summary_func = summary_func
        self.grps = None

    def fit(self, df):
        """
        Fitting

        :param df: Pandas DataFrame
        """

        self.grps = df.groupby(self.key).agg({self.target: lambda x: self.summary_func(x)}).to_dict()[self.target]

        return self

    def __repr__(self):
        return f"Variable Transformer (key={self.key}, target={self.target}, strategy={self.strategy})"

    def transform(self, df):
        """
        Transforming

        :param df: Pandas DataFrame
        """
        if self.key not in df.columns or self.target not in df.columns:
            raise ValueError(f"Either {self.key} or {self.target} not in {df.columns}")

        grps = self.grps
        if self.strategy == "zscore":
            X = df[[self.key, self.target]].apply(
                lambda x: (x[1] - grps[x[0]]["mean_val"]) / grps[x[0]]["std_val"], axis=1
            )
        elif self.strategy == "min-max":
            X = df[[self.key, self.target]].apply(
                lambda x: (x[1] - grps[x[0]]["min_val"]) / grps[x[0]]["range_val"], axis=1
            )
        elif self.strategy == "median-iqr":
            X = df[[self.key, self.target]].apply(
                lambda x: (x[1] - grps[x[0]]["median_val"]) / grps[x[0]]["iqr_val"], axis=1
            )
        elif self.strategy == "mean-iqr":
            X = df[[self.key, self.target]].apply(
                lambda x: (x[1] - grps[x[0]]["mean_val"]) / grps[x[0]]["iqr_val"], axis=1
            )
        elif self.strategy == "median-qd":
            X = df[[self.key, self.target]].apply(
                lambda x: (x[1] - grps[x[0]]["median_val"]) / grps[x[0]]["qd_val"], axis=1
            )
        elif self.strategy == "median-std":
            X = df[[self.key, self.target]].apply(
                lambda x: (x[1] - grps[x[0]]["median_val"]) / grps[x[0]]["std_val"], axis=1
            )
        elif self.strategy == "median-range":
            X = df[[self.key, self.target]].apply(
                lambda x: (x[1] - grps[x[0]]["median_val"]) / grps[x[0]]["range_val"], axis=1
            )
        elif self.strategy == "mean-range":
            X = df[[self.key, self.target]].apply(
                lambda x: (x[1] - grps[x[0]]["mean_val"]) / grps[x[0]]["range_val"], axis=1
            )
        elif self.strategy == "min-qd":
            X = df[[self.key, self.target]].apply(
                lambda x: (x[1] - grps[x[0]]["min_val"]) / grps[x[0]]["qd_val"], axis=1
            )
        elif self.strategy == "min-iqr":
            X = df[[self.key, self.target]].apply(
                lambda x: (x[1] - grps[x[0]]["min_val"]) / grps[x[0]]["iqr_val"], axis=1
            )
        elif self.strategy == "min":
            X = df[[self.key, self.target]].apply(lambda x: x[1] / grps[x[0]]["min_val"], axis=1)
        elif self.strategy == "max":
            X = df[[self.key, self.target]].apply(lambda x: x[1] / grps[x[0]]["max_val"], axis=1)
        elif self.strategy == "mean":
            X = df[[self.key, self.target]].apply(lambda x: x[1] / grps[x[0]]["mean_val"], axis=1)
        elif self.strategy == "median":
            X = df[[self.key, self.target]].apply(lambda x: x[1] / grps[x[0]]["median_val"], axis=1)
        elif self.strategy == "iqr":
            X = df[[self.key, self.target]].apply(lambda x: x[1] / grps[x[0]]["iqr_val"], axis=1)
        elif self.strategy == "qd":
            X = df[[self.key, self.target]].apply(lambda x: x[1] / grps[x[0]]["qd_val"], axis=1)
        elif self.strategy == "std":
            X = df[[self.key, self.target]].apply(lambda x: x[1] / grps[x[0]]["std_val"], axis=1)
        elif self.strategy == "range":
            X = df[[self.key, self.target]].apply(lambda x: x[1] / grps[x[0]]["range_val"], axis=1)
        elif self.strategy == "custom" and self.custom_func is not None:
            X = df[[self.key, self.target]].apply(lambda x: self.custom_func(x[0], x[1], grps, inverse=False), axis=1)

        return X

    def inverse_transform(self, df, target=None):

        if target is None:
            target = self.target
        if self.key not in df.columns or target not in df.columns:
            raise ValueError(f"Either {self.key} or {target} not in {df.columns}")

        grps = self.grps
        if self.strategy == "zscore":
            X = df[[self.key, target]].apply(lambda x: (x[1] * grps[x[0]]["std_val"] + grps[x[0]]["mean_val"]), axis=1)
        elif self.strategy == "min-max":
            X = df[[self.key, target]].apply(lambda x: (x[1] * grps[x[0]]["range_val"] + grps[x[0]]["min_val"]), axis=1)
        elif self.strategy == "median-iqr":
            X = df[[self.key, target]].apply(
                lambda x: (x[1] * grps[x[0]]["iqr_val"] + grps[x[0]]["median_val"]), axis=1
            )
        elif self.strategy == "mean-iqr":
            X = df[[self.key, target]].apply(lambda x: (x[1] * grps[x[0]]["iqr_val"] + grps[x[0]]["mean_val"]), axis=1)
        elif self.strategy == "median-qd":
            X = df[[self.key, target]].apply(lambda x: (x[1] * grps[x[0]]["qd_val"] + grps[x[0]]["median_val"]), axis=1)
        elif self.strategy == "median-std":
            X = df[[self.key, target]].apply(
                lambda x: (x[1] * grps[x[0]]["std_val"] + grps[x[0]]["median_val"]), axis=1
            )
        elif self.strategy == "median-range":
            X = df[[self.key, target]].apply(
                lambda x: (x[1] * grps[x[0]]["range_val"] + grps[x[0]]["median_val"]), axis=1
            )
        elif self.strategy == "mean-range":
            X = df[[self.key, target]].apply(
                lambda x: (x[1] * grps[x[0]]["range_val"] + grps[x[0]]["mean_val"]), axis=1
            )
        elif self.strategy == "min-qd":
            X = df[[self.key, target]].apply(lambda x: (x[1] * grps[x[0]]["qd_val"] + grps[x[0]]["min_val"]), axis=1)
        elif self.strategy == "min-iqr":
            X = df[[self.key, target]].apply(lambda x: x[1] * grps[x[0]]["iqr_val"] + grps[x[0]]["min_val"], axis=1)
        elif self.strategy == "min":
            X = df[[self.key, target]].apply(lambda x: x[1] * grps[x[0]]["min_val"], axis=1)
        elif self.strategy == "max":
            X = df[[self.key, target]].apply(lambda x: x[1] * grps[x[0]]["max_val"], axis=1)
        elif self.strategy == "mean":
            X = df[[self.key, target]].apply(lambda x: x[1] * grps[x[0]]["mean_val"], axis=1)
        elif self.strategy == "median":
            X = df[[self.key, target]].apply(lambda x: x[1] * grps[x[0]]["median_val"], axis=1)
        elif self.strategy == "iqr":
            X = df[[self.key, target]].apply(lambda x: x[1] * grps[x[0]]["iqr_val"], axis=1)
        elif self.strategy == "qd":
            X = df[[self.key, target]].apply(lambda x: x[1] * grps[x[0]]["qd_val"], axis=1)
        elif self.strategy == "std":
            X = df[[self.key, target]].apply(lambda x: x[1] * grps[x[0]]["std_val"], axis=1)
        elif self.strategy == "range":
            X = df[[self.key, target]].apply(lambda x: x[1] * grps[x[0]]["range_val"], axis=1)
        elif self.strategy == "custom" and self.custom_func is not None:
            X = df[[self.key, target]].apply(lambda x: self.custom_func(x[0], x[1], grps, inverse=True), axis=1)

        return X


## todo: Trended variable transformation the means and all would be treand at level of granularity of
#        time (year,year month, year-week , month-week and so) given by user.
## todo: Single Ungrouped Variable transformation
## todo: Decomposed (series -trend -seasonality) based variabel transformation


## Multiple label Encoding with easy transforms
class IndexMapper(object):

    """

        A multi column categorical labeller

    :param categorical_columns: Columns to create Label for
            :param verbose: Verbosity
    """

    def __init__(self, categorical_columns, verbose=False):

        self.indexes = {}
        self.cat_cols = categorical_columns
        self.verbose = verbose

    def _idx_generator(self, codes):
        i = 0
        codes2idx = {}
        idx2codes = {}
        for k in codes:
            if k not in codes2idx.keys():
                codes2idx[k] = i
                idx2codes[i] = k
            i += 1
        return codes2idx, idx2codes

    def fit(self, df, y=None):
        """
        Fitting

        :param df: Pandas DataFrame
        """

        if np.any([col not in df.columns for col in self.cat_cols]):
            raise ValueError(f"Not all categorical columns {self.cat_cols} is in data given {df.columns.tolist()}")

        for col in self.cat_cols:
            if self.verbose:
                print("Creating mapping for :", col)
            self._make_index(df, col)

    def transform(self, df, inplace=False):
        """
        Transforming

        :param df: Pandas DataFrame
        """

        if not inplace:
            sub_df = []
        for col in self.cat_cols:
            if self.verbose:
                print("Mapping Columns for :", col)

            if inplace:
                df[col] = self._replace_code(df, col)
            else:
                sub_df.append(self._replace_code(df, col))

        return df if inplace else pd.concat(sub_df, axis=1)

    def inverse_transform(self, df, inplace=False):
        if not inplace:
            sub_df = []
        for col in self.cat_cols:
            if self.verbose:
                print("Inverse Mapping Columns for :", col)

            if inplace:
                df[col] = self._replace_code(df, col)
            else:
                sub_df.append(self._replace_index(df, col))

        return df if inplace else pd.concat(sub_df, axis=1)

    def _make_index(self, df, name):
        if name in self.indexes.keys():
            print("Replacing original exsisting mapping for {} with new data".format(name))
            print(self.indexes.pop(name))

        codes, indexes = self._idx_generator(df[name].unique().tolist())
        self.indexes[name] = {"codes": codes, "index": indexes}

    def _replace_code(self, df, name):
        if name not in self.indexes.keys():
            self.make_index(df, name)
        code = (
            df[name]
            .apply(lambda x: self.indexes[name]["codes"][x])
            .astype(np.uint16 if len(self.indexes[name]["codes"]) > 255 else np.uint8)
        )
        return code

    def get_all_names(self, name):
        if name not in self.indexes.keys():
            raise ValueError("'{}' not in indexed yet!".format(name))
        return list(self.indexes[name]["codes"].keys())

    def get_all_index(self, name):
        if name not in self.indexes.keys():
            raise ValueError("'{}' not in indexed yet!".format(name))
        return list(self.indexes[name]["index"].keys())

    def get_name(self, name, idx):
        if name not in self.indexes.keys():
            raise ValueError("'{}' not in indexed yet!".format(name))
        return self.indexes[name]["index"][idx]

    def get_index(self, name, value):
        if name not in self.indexes.keys():
            raise ValueError("'{}' not in indexed yet!".format(name))
        return self.indexes[name]["codes"][value]

    def _replace_index(self, df, name):
        if name not in self.indexes.keys():
            raise ValueError("'{}' not indexed!".format(name))
        code = df[name].apply(lambda x: self.indexes[name]["index"][x])
        return code
