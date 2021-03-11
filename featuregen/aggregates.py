# basic imports
import numpy as np
import pandas as pd
from dateutil.relativedelta import SU, TH, relativedelta
from sklearn.base import BaseEstimator, TransformerMixin, clone

# local imports
from .utils import check_values_in_array


class AggregateFeatures(BaseEstimator, TransformerMixin):

    """

    Time Based Feature Aggregation for groups

    :param time_key: Date / Time column to used for creating aggregation
    :param key: Grouping / ID column against which aggregatoin would happen
    :param target: Target column name which needs to be used for creating aggregation
    :param start_date: Start Date to create the Features
    :param end_date: End Date to create the Features
    :param time_aggregates: Time Granularity
    :param hemisphere: Sub Levels to create encodings
    :param monotone_constraints: Montonic Constraints

    """

    def __init__(
        self,
        time_key,
        key,
        target,
        start_date=None,
        end_date=None,
        time_aggregates=[
            "year",
            "halfyear",
            "season",
            "quarter",
            "month",
            "week",
            "year_month",
            "year_weekday",
            "halfyear_weekday",
            "season_weekday",
            "quarter_weekday",
            "month_weekday",
            "weekday",
        ],
        hemisphere="north",
        monotone_constraints=False,
    ):

        self.target = target
        self.key = key
        self.time_key = time_key

        # understand impact of start and end dates
        self.start_date = start_date if start_date is not None else "2019-01-01"
        self.end_date = end_date if end_date is not None else "2019-12-31"

        self.time_aggregates = time_aggregates
        # self.num_seasons = num_seasons
        self.hemisphere = hemisphere
        self.flag = "predict"
        self.monotone_constraints = monotone_constraints
        self.constraints = {}
        self.data = None

        # # something for season, just a place holder as of now
        # df['date_offset'] = (df.date_UTC.dt.month*100 + df.date_UTC.dt.day - 320)%1300

        # df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300],
        #               labels=['spring', 'summer', 'autumn', 'winter']

    def _get_season(self, month, day):

        md = month * 100 + day

        if (md > 320) and (md < 621):
            s = 0  # spring
        elif (md > 620) and (md < 923):
            s = 1  # summer
        elif (md > 922) and (md < 1223):
            s = 2  # fall
        else:
            s = 3  # winter

        if not self.hemisphere == "north":
            s = (s + 2) % 3
        return s

    def _get_dim(self, actual_data, dims):

        reference_data = self.data.copy()
        added_col, group_cols = [], []

        for dim in dims.split("_"):
            if dim == "year":
                actual_data["year"] = actual_data[self.time_key].dt.year

            elif dim == "halfyear":
                actual_data["halfyear"] = 1 * (actual_data[self.time_key].dt.month > 6)

            elif dim == "season":
                actual_data["season"] = actual_data[self.time_key].apply(lambda x: self._get_season(x.month, x.day))

            elif dim == "quarter":
                actual_data["quarter"] = "quarter_" + actual_data[self.time_key].dt.quarter.astype("str")

            elif dim == "month":
                actual_data["month"] = actual_data[self.time_key].dt.month_name()

            elif dim == "week":
                actual_data["week"] = "week_" + actual_data[self.time_key].dt.weekofyear.astype("str")

            elif dim == "weekday":
                actual_data["weekday"] = actual_data[self.time_key].dt.day_name()

            else:
                return actual_data, None

            added_col.append(dim)
            group_cols.append(dim)

        # # defining data to be grouped on
        # agg = data[(data['date'] >= start_date) & (data['date'] <= end_date)] if flag != 'predict' else self.data

        # # definign the grouping columns,
        # group_cols = ['month', 'weekday', self.key] if dim == 'weekday' else [dim, self.key]

        # print("Added Columns     : ",added_col)
        # print("Grouping Columns  : ",group_cols)
        # print("Reference Columns : ",reference_data.columns)
        # print("Actual Columns    : ",actual_data.columns)

        # doing group by and renaming appropriately
        group_cols.append(self.key)
        agg = reference_data.groupby(group_cols).mean()[self.target].reset_index()
        var_name = self.target + "_" + dims
        agg.rename(columns={self.target: var_name}, inplace=True)

        #         print('Added Cols:',added_col,' grouping cols:',group_cols)
        #         print(agg.head())
        if self.monotone_constraints:
            if var_name not in self.constraints.keys():
                self.constraints[var_name] = 1

        # merging with data
        agg = pd.merge(actual_data, agg, on=group_cols, how="left")
        agg.drop(columns=added_col, axis=1, inplace=True)

        return agg, var_name

    def fit(self, X, y=None):
        """

        Assumption before fetching data in fit:
        1) Time columns is named as date and it is in datetime format of pandas
        2) Grouper and Target column are present in data
        3) There are no NAs as of now and all preprocesing to this is done in prior

        :param X: pandas Data Frame
        """

        nas = X.isna().sum(axis=0)
        if np.any(nas):
            raise ValueError("Data Frame with NAs is not allowed to be processed. Please the data. ", nas)

        cols = X.columns
        if not check_values_in_array([self.time_key, self.key, self.target], cols):
            raise ValueError(
                "Either {} or {} or {} not available in {} ".format(self.time_key, self.key, self.target, X.columns)
            )

        # assigning data
        self.data = X[(X[self.time_key] >= self.start_date) & (X[self.time_key] <= self.end_date)]
        self.data = self.data[[self.time_key, self.key, self.target]].copy(deep=True)
        self.data.drop_duplicates([self.time_key, self.key], inplace=True)

        # Addign features
        self.data["year"] = self.data[self.time_key].dt.year
        self.data["halfyear"] = 1 * (self.data[self.time_key].dt.month > 6)
        self.data["season"] = self.data[self.time_key].apply(lambda x: self._get_season(x.month, x.day))
        self.data["quarter"] = "quarter_" + self.data[self.time_key].dt.quarter.astype("str")
        self.data["month"] = self.data[self.time_key].dt.month_name()
        self.data["week"] = "week_" + self.data[self.time_key].dt.weekofyear.astype("str")
        self.data["weekday"] = self.data[self.time_key].dt.day_name()

        print("Agreagate Features would build from \nShape: {} Columns:{}".format(self.data.columns, self.data.shape))
        self.features = self.data.columns.tolist()
        return self

    def transform(self, X, y=None, flag="predict", start_date=None, end_date=None, verbose=False):
        """
        Transforming

        :param X: Pandas DataFrame
        """

        added_features = []
        for dim in self.time_aggregates:
            X, var_name = self._get_dim(X, dim)
            added_features.append(var_name)

        # X = X[~X.isnull().any(1)]
        print("Added Features : ", added_features)
        print("Post Transforming with Aggregate Features data shape is:", X.shape)
        if start_date != None:
            X = X[(X[self.time_key] >= start_date) & (X[self.time_key] <= end_date)]
        return X


# Features which are smaller thatn the key but are not directly mapped in data
class SubLevelFeatures(BaseEstimator, TransformerMixin):

    """

    Group wise Sub-Group Level features acting as embeddings for groups.

    :param date_key: Date / Time column to used for creating aggregation
    :param key: Grouping / ID column against which aggregatoin would happen
    :param target: Target column name which needs to be used for creating aggregation
    :param start_date: Start Date to create the Features
    :param end_date: End Date to create the Features
    :param granularity: Time Granularity
    :param sub_levels: Sub Levels to create encodings
    :param factor_cutoff: Factor Cutoff
    :param encode_type: Encoding
    :param monotone_constraints: Montonic Constraints

    """

    def __init__(
        self,
        date_key,
        key,
        target=None,
        start_date="2019-01-01",
        end_date="2019-12-31",
        granularity="yearly",
        sub_levels=["size"],
        factor_cutoff=0.003,
        encode_type="none",
        monotone_constraints=False,
    ):

        self.target = target
        self.key = key

        # understand impact of start and end dates
        # this would be used in case we are using higher hierarchy in play
        self.start_date = start_date if start_date is not None else "2019-01-01"
        self.end_date = end_date if end_date is not None else "2019-12-31"

        # granularity of these levels
        self.granularity = granularity

        # product features to extract
        self.sub_levels = sub_levels
        self.factor_cutoff = factor_cutoff
        self.encode_type = "one-hot"  # only option as of now
        self.monotone_constraints = monotone_constraints
        self.constraints = {}
        self.data = None

    def _get_pct_by_class(self, df, dim):
        c = df.groupby([self.key, dim])["reference"].sum().rename("count")
        # # (c / c.groupby(level=[0]).transform("sum")).reset_index()
        df_pct = pd.pivot(
            (c / c.groupby(level=[0]).transform("sum")).reset_index(), index=self.key, columns=dim, values="count"
        ).fillna(0)
        df_pct.columns = [
            dim + "_" + str(c).strip().lower().replace("/", "").replace(" ", "_") for c in df_pct.columns.ravel()
        ]
        return df_pct.reset_index()

    def fit(self, X=None, y=None):

        """
        :param X: Pandas DataFrame
        """

        df = X.copy(deep=True)
        df["reference"] = 1

        # creating the product features above the key value:
        fin_df = (
            df.groupby(["category_desc", "dept_desc", self.key], as_index=False)["reference"]
            .sum()
            .drop("reference", axis=1)
        )

        ## adding percentage components for each class
        for dim in self.product_levels:
            fin_df = pd.merge(fin_df, self._get_pct_by_class(df, dim), on=self.key, how="outer")
        fin_df.fillna(0, inplace=True)
        print("Product Features Created with shape {}".format(fin_df.shape))
        self.data = fin_df
        self.features = self.data.columns.tolist()
        if self.monotone_constraints:
            #             for col in slef.product_levels:
            for feat in self.features:
                if feat not in self.constraints:
                    if any([feat.startswith(col) for col in self.product_levels]):
                        self.constraints[feat] = 1
                    elif feat not in [self.key, self.time_key, self.target]:
                        self.constraints[feat] = 0

        return self

    def transform(self, X, y=None, flag="predict"):
        """
        Transforming

        :param X: Pandas DataFrame
        """

        if self.key not in X.columns:
            raise ValueError("{} not found in input data. Columns {}".format(self.key, X.columns))

        X = pd.merge(X, self.data, on=self.key, how="left")
        print("Added Features : ", [feat for feat in self.features if feat != self.key])
        print("Post Transforming with Product Features data shape is:", X.shape)

        return X
