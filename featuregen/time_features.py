"""
File for generating and dealing with Holiday and time related features
"""

# base imports
import copy
from datetime import datetime, time, timedelta
import numpy as np
import pandas as pd
from dateutil.relativedelta import MO, SU, TH, relativedelta
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import Day, Easter
from sklearn.base import BaseEstimator, TransformerMixin, clone

# local imports
from .utils import check_values_in_array


## Lag features generation
class LagFeatures(BaseEstimator, TransformerMixin):

    """

    Lags for Target variable Of Groups

    :param time_key: Date / Time column to used for creating aggregation
    :param key: Grouping / ID column against which aggregatoin would happen
    :param target: Target column name which needs to be used for creating aggregation
    :param start_date: Start Date to create the Features
    :param end_date: End Date to create the Features
    :param lags: list of lags to be created
    :param monotone_constraints: Montonic Constraint
    """

    def __init__(self, time_key, target, key, start_date=None, end_date=None, lags=None, monotone_constraints=False):

        self.target = target
        self.key = key
        self.time_key = time_key

        # understand impact of start and end dates
        self.start_date = start_date if start_date is not None else "2016-01-01"
        self.end_date = end_date if end_date is not None else "2019-12-31"

        self.lags = lags
        self.flag = "predict"
        self.monotone_constraints = monotone_constraints
        self.constraints = {}
        self.data = None

    def _get_lag(self, X, lag):
        X[self.time_key] = pd.to_datetime(X[self.time_key])
        rel_lag_var = self.time_key + "_lag" + str(lag)
        X[rel_lag_var] = [date - relativedelta(days=lag) for date in X[self.time_key]]
        df = pd.merge(
            X,
            self.data[[self.time_key, self.key, self.target]],
            how="left",
            left_on=[rel_lag_var, self.key],
            right_on=[self.time_key, self.key],
        ).copy()

        df.rename(
            columns={
                self.time_key + "_x": self.time_key,
                self.target + "_x": self.target,
                self.target + "_y": "lag_" + str(lag),
                self.target: "lag_" + str(lag),
            },
            inplace=True,
        )

        df.drop(columns=[self.time_key + "_y", rel_lag_var], inplace=True)

        return df

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
            raise ValueError("Either date or {} or {} not available in {} ".format(self.key, self.target, X.columns))

        # assigning data
        self.data = X[[self.time_key, self.key, self.target]].copy(deep=True)
        self.data.drop_duplicates([self.time_key, self.key], inplace=True)

        print("Lag Features would build from \nShape: {} Columns:{}".format(self.data.columns, self.data.shape))

        self.features = self.data.columns.tolist()
        if self.monotone_constraints:
            for feat in self.features:
                if feat.startswith("lag"):
                    self.constraints[feat] = 1

        return self

    def transform(self, X, y=None, flag="predict", start_date=None, end_date=None):
        """
        Transforming

        :param X: Pandas DataFrame
        """

        lags_features = []
        if self.lags is not None and isinstance(self.lags, list):
            for lag in self.lags:
                lags_features.append("lag_" + str(lag))
                X = self._get_lag(copy.deepcopy(X), lag=int(lag))
        else:
            X = self._get_lag(copy.deepcopy(X), lag=14)
            lags_features.append("lag_" + str(14))
            X = self._get_lag(copy.deepcopy(X), lag=7)
            lags_features.append("lag_" + str(7))

        print("Added Features : ", lags_features)
        print("Post Transforming with Lag Features data shape is:", X.shape)

        return X


# processing Holiday / Main Events Features
class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("new_years_day", month=1, day=1),
        Holiday("mlk_day", month=1, day=1, offset=pd.DateOffset(weekday=MO(3))),
        Holiday("super_bowl", month=2, day=1, offset=pd.DateOffset(weekday=SU(1))),
        Holiday("valentines_day", month=2, day=14),
        Holiday("presidents_day", month=2, day=1, offset=pd.DateOffset(weekday=MO(3))),
        Holiday("easter", month=1, day=1, offset=[Easter()]),
        Holiday("mothers_day", month=5, day=1, offset=pd.DateOffset(weekday=SU(2))),
        Holiday("memorial_day", month=5, day=31, offset=pd.DateOffset(weekday=MO(-1))),
        Holiday("july_4th", month=7, day=4),
        Holiday("labor_day", month=9, day=1, offset=pd.DateOffset(weekday=MO(1))),
        Holiday("columbus_day", month=10, day=1, offset=pd.DateOffset(weekday=MO(2))),
        Holiday("halloween", month=10, day=31),
        Holiday("veterans_day", month=11, day=11),
        Holiday("thanksgiving", month=11, day=1, offset=pd.DateOffset(weekday=TH(4))),
        Holiday("black_friday", month=11, day=1, offset=[pd.DateOffset(weekday=TH(4)), Day(1)]),
        Holiday("cyber_monday", month=11, day=1, offset=[pd.DateOffset(weekday=TH(4)), Day(4)]),
        Holiday("christmas", month=12, day=25),
    ]


class HolidayAndEventFeatures(BaseEstimator, TransformerMixin):

    """

    Time Based Feature Aggregation for groups

    :param time_key: Date / Time column to used for creating aggregation
    :param start_date: Start Date to create the Features
    :param holidays: List of all holidays to be created
    :param impact_window: To create holiday ipact range or not
    :param lower_window: Lower limit of impact range
    :param upper_window: Upper limit of impact range
    :param expand_holiday: Expand Holidays to one hot columns
    :param expand_lower: Expand Lower impact range to one hot columns
    :param expand_upper: Expand Upper impcat range  to one hot columns
    :param indicator: Indicator as 'value'
    :param verbose: Verbosity
    :param monotone_constraints: Montonic Constraints

    """

    def __init__(
        self,
        time_key,
        start_date=None,
        end_date=None,
        holidays="all",
        impact_window=True,
        lower_window=-7,
        upper_window=7,
        expand_holiday=False,
        expand_lower=False,
        expand_upper=False,
        indicator="value",
        monotone_constraints=False,
        verbose=False,
    ):

        self.all_holidays = [
            "new_years_day",
            "mlk_day",
            "super_bowl",
            "valentines_day",
            "presidents_day",
            "easter",
            "mothers_day",
            "memorial_day",
            "july_4th",
            "prime_day",
            "labor_day",
            "columbus_day",
            "halloween",
            "veterans_day",
            "thanksgiving",
            "black_friday",
            "cyber_monday",
            "christmas",
        ]

        self.time_key = time_key
        self.start_date = start_date
        self.end_date = end_date
        if holidays == "all":
            self.holidays = self.all_holidays
        else:
            self.holidays = [h for h in self.all_holidays if h in holidays]
            print("Final Holidays to be processed:", self.holidays)
        self.impact_window = impact_window
        self.expand_holiday = expand_holiday
        self.expand_lower = expand_lower
        self.expand_upper = expand_upper
        self.lower_window = lower_window
        self.upper_window = upper_window
        self.indicator = indicator
        self.features = None
        self.input = None
        self.data = None
        self.monotone_constraints = monotone_constraints
        self.constraints = {}
        self.verbose = verbose
        self.flag = "predict"

    def _getHolidays(self):

        # List of holidays to track and years when they occur
        holidayList = [h for h in self.all_holidays if h != "prime_day"]

        years = [2017, 2018, 2019, 2020, 2021]

        for i in range(len(years)):
            tmpHol = pd.DataFrame(
                {
                    self.time_key: USTradingCalendar().holidays(datetime(years[i], 1, 1), datetime(years[i], 12, 31)),
                    "holiday": holidayList,
                }
            )
            if i == 0:
                historicalHol = tmpHol
            else:
                historicalHol = historicalHol.append(tmpHol)

        # adding prime days
        historicalHol = historicalHol.append(
            pd.DataFrame({self.time_key: datetime(2016, 7, 12), "holiday": "prime_day"}, index=[0])
        )
        historicalHol = historicalHol.append(
            pd.DataFrame({self.time_key: datetime(2017, 7, 10), "holiday": "prime_day"}, index=[0])
        )
        historicalHol = historicalHol.append(
            pd.DataFrame({self.time_key: datetime(2018, 7, 16), "holiday": "prime_day"}, index=[0])
        )
        historicalHol = historicalHol.append(
            pd.DataFrame({self.time_key: datetime(2019, 7, 15), "holiday": "prime_day"}, index=[0])
        )
        historicalHol = historicalHol.append(
            pd.DataFrame({self.time_key: datetime(2020, 10, 13), "holiday": "prime_day"}, index=[0])
        )

        # to process only selected holidays:
        historicalHol = historicalHol[historicalHol.holiday.isin(self.holidays)].copy()

        # sortting the holida
        historicalHol = historicalHol.sort_values(["holiday", self.time_key]).reset_index(drop=True)
        return historicalHol

    def _add_holiday_columns(self, holidayTbl, df):

        # default_lower_window = self.lower_window
        # default_upper_window = self.upper_window
        holidayList = np.unique(holidayTbl.holiday)
        holiday_vector = None

        if self.verbose:
            print("Holiday To Process :", holidayList)
        for i in range(len(holidayList)):
            holiday_name = holidayList[i]
            addHoliday = holidayTbl[holidayTbl.holiday == holiday_name]

            # print(addHoliday)

            lower_window = self.lower_window
            upper_window = self.upper_window
            if holiday_name == "cyber_monday":
                lower_window = -1 if lower_window < -1 else lower_window
            if holiday_name == "mothers_day":
                lower_window = -5 if lower_window < -5 else lower_window
            if holiday_name == "christmas":
                lower_window = -5 if lower_window < -5 else lower_window
            if holiday_name == "memorial_day":
                lower_window = -5 if lower_window < -5 else lower_window
            if holiday_name == "easter":
                lower_window = -5 if lower_window < -5 else lower_window
            if holiday_name == "prime_day":
                upper_window = 1 if upper_window > 1 else upper_window

            # criteria where we need impact windows as a single column for a holiday
            if self.impact_window and not self.expand_lower and not self.expand_upper:
                flagRange = range(lower_window, upper_window + 1)
                holiday_impact_range = []
                for holiday_date in addHoliday.date:
                    holiday_impact_range = holiday_impact_range + [
                        holiday_date + timedelta(days=flg) for flg in flagRange
                    ]

            # criteria wher we want to one hot upper windows only
            elif self.impact_window and self.expand_upper:
                flagRange = range(lower_window, 0)
                holiday_impact_range = [hl_date for hl_date in addHoliday.date]
                for holiday_date in addHoliday.date:
                    holiday_impact_range = holiday_impact_range + [
                        holiday_date + timedelta(days=flg) for flg in flagRange
                    ]

            # criteria wher we want to one hot lower windows only
            elif self.impact_window and self.expand_lower:
                flagRange = range(1, upper_window + 1)
                holiday_impact_range = [hl_date for hl_date in addHoliday.date]
                for holiday_date in addHoliday.date:
                    holiday_impact_range = holiday_impact_range + [
                        holiday_date + timedelta(days=flg) for flg in flagRange
                    ]

            # criteria wher we want to one hot both windows
            elif not self.impact_window:
                holiday_impact_range = [hl_date for hl_date in addHoliday.date]

            if self.verbose:
                print("Holiday: ", holiday_name)
                print("Holiday Impact Range: ", len(holiday_impact_range), holiday_impact_range)

            # do we want a sprate column for each holiday
            if self.verbose:
                print("Creating Holiday Feature")
            if self.expand_holiday:
                df[holiday_name] = 1 * (df.date.isin(holiday_impact_range))
            else:
                hl_vector = 1 * (df.date.isin(holiday_impact_range))
                holiday_vector = holiday_vector + hl_vector if holiday_vector is not None else hl_vector

            # one hot encoding upper windows
            if self.verbose:
                print("Creating Upper Impact Zone")
            if self.impact_window and self.expand_upper:
                for flg in range(1, upper_window + 1):
                    idenDts = addHoliday.date + timedelta(days=flg)
                    col_name = holiday_name + "_" + str(flg) if self.expand_holiday else "upper_" + str(flg)

                    if col_name in df.columns:
                        df[col_name] = df[col_name] + (df.date.isin(idenDts) * 1)
                    else:
                        df[col_name] = df.date.isin(idenDts) * 1

            # one hot encoding lower windows
            if self.verbose:
                print("Creating Lower Impact Zone")
            if self.impact_window and self.expand_lower:
                for flg in range(lower_window, 0):
                    idenDts = addHoliday.date + timedelta(days=flg)
                    col_name = holiday_name + "_" + str(flg) if self.expand_holiday else "lower_" + str(flg)

                    if col_name in df.columns:
                        df[col_name] = df[col_name] + (df.date.isin(idenDts) * 1)
                    else:
                        df[col_name] = df.date.isin(idenDts) * 1

            # if(len(flagRange) > 0):
            #     for f in flagRange:
            #         idenDts = addHoliday.date + timedelta(days = flagRange[f])
            #         df[str(addHoliday.holiday.iloc[0])+'_'+str(flagRange[f])] = df.date.isin(idenDts).astype('float')
            # else:
            #     idenDts = addHoliday.date
            #     df[str(addHoliday.holiday.iloc[0])+'_0'] = df.date.isin(idenDts).astype('float')

        if not self.expand_holiday:
            df["holidays"] = holiday_vector

        return df

    def _add_days_wrt_holiday(self, date_list, holidayTbl, direction="since"):

        dayVec = []

        # todo: have to rewrite this w.r.t time key
        for i in range(date_list.shape[0]):
            if direction == "since":
                calc = (date_list.date[i] - holidayTbl[self.time_key]) / np.timedelta64(1, "D")
            elif direction == "until":
                calc = (holidayTbl[self.time_key] - date_list.date[i]) / np.timedelta64(1, "D")
            daysSince = min(calc[calc > 0])
            dayVec.append(daysSince)

        return dayVec

    def fit(self, X=None, y=None):
        """
        Fitting

        :param X: None and has no impact
        """

        df = pd.DataFrame({self.time_key: pd.date_range(start=self.start_date, end=self.end_date)})
        print(
            "Dates Created with \nShape :{} Min Date: {} Max Date: {}".format(
                df.shape, df[self.time_key].min(), df[self.time_key].max()
            )
        )

        # Holiday table
        holidays = self._getHolidays()

        # Add holiday based signalling features
        df = self._add_holiday_columns(holidayTbl=holidays, df=df)

        # Add holiday columns and timing features
        dfDates = pd.DataFrame({self.time_key: df[self.time_key].unique()})
        dfDates["days_since_last_hol"] = self._add_days_wrt_holiday(
            date_list=dfDates, holidayTbl=holidays, direction="since"
        )
        dfDates["days_until_next_hol"] = self._add_days_wrt_holiday(
            date_list=dfDates, holidayTbl=holidays, direction="until"
        )

        df = pd.merge(df, dfDates, on=self.time_key, how="left")

        # Add store closed for holiday indicator
        addHoliday = holidays[
            (holidays.holiday == "easter") | (holidays.holiday == "thanksgiving") | (holidays.holiday == "christmas")
        ]
        df["closed_for_holiday_ind"] = 1 * (
            df[self.time_key].isin(addHoliday[self.time_key].dt.date.astype("str").values.tolist())
        )

        self.data = df
        self.features = self.data.columns.tolist()
        if self.monotone_constraints:
            for feat in self.features:
                if feat != self.time_key:
                    self.constraints[feat] = 1

        return self

    def transform(self, X, start_date=None, end_date=None):
        """
        Transforming

        :param X: Pandas DataFrame
        """
        if self.time_key not in X.columns:
            raise ValueError("{} not found in input data. Columns {}".format(self.time_key, X.columns))

        data = self.data.copy(deep=True)
        # X[self.time_key] = pd.to_datetime(X[self.time_key])

        if start_date != None:
            X = X[(X[self.time_key] >= start_date) & (X[self.time_key] <= end_date)]

        X = pd.merge(X, data, on=self.time_key, how="left")
        print("Added Features : ", [feat for feat in self.features if feat != self.time_key])
        print("Post Transforming with Holiday Features data shape is:", X.shape)

        return X


#         if self.flag == 'predict':
#             return
#         else:
#             data = data[(data[self.time_key] >= start_date) & (data[self.time_key] <= end_date)]
#             return pd.merge(data, X, on=self.time_key, how='left')


## Calendar features
class CalendarFeatures(BaseEstimator, TransformerMixin):

    """

    Calendar features based on dates

    :param time_key: Date / Time column to used for creating aggregation
    :param key: Grouping / ID column against which aggregatoin would happen
    :param target: Target column name which needs to be used for creating aggregation
    :param start_date: Start Date to create the Features
    :param end_date: End Date to create the Features
    :param expand_days: expand the days to one hot or not
    :param monotone_constraints: Montonic Constraints

    """

    def __init__(
        self, key, time_key, target=None, start_date=None, end_date=None, expand_days=False, monotone_constraints=False
    ):

        self.start_date = start_date
        self.end_date = end_date
        self.key = key
        self.target = target
        self.time_key = time_key
        self.expand_days = expand_days
        self.monotone_constraints = monotone_constraints
        self.constraints = {}
        self.data = None
        self.features = None

    def _load_calendar(self, dataset):

        dates = dataset[self.time_key].copy()

        # Substring week and month columns
        dates["day"] = dates[self.time_key]
        dates["month_day"] = dates[self.time_key]
        dates["month"] = dates[self.time_key].dt.quarter.astype("str")
        dates["quarter_day"] = dates[self.time_key].dt.quarter.astype("str")
        dates["quarter"] = dates[self.time_key].dt.quarter.astype("str")
        dates["season"] = dates[self.time_key]
        dates["season_day"] = dates[self.time_key].dt
        dates["year"] = dates[self.time_key].dt.year

        return dates

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

    # todo: avery imprtant update
    def fit(self, X=None, y=None):
        """
        Fitting

        :param X: None and has no impact
        """

        df = pd.DataFrame({self.time_key: pd.date_range(start=self.start_date, end=self.end_date)})
        print(
            "Dates Created with \nShape :{} Min Date: {} Max Date: {}".format(
                df.shape, df[self.time_key].min(), df[self.time_key].max()
            )
        )

        # Fiscal calendar
        cal = self._load_calendar(X)
        df = pd.merge(df, cal, on=self.time_key, how="left")
        num_cols = ["day", "month_day", "month", "quarter_day", "quarter", "season_day", "season", "year"]
        df[num_cols] = df[num_cols].apply(lambda x: x.astype(int))
        df["season"] = df[self.time_key].apply(lambda x: self._get_season(x.month, x.day))
        print("Date Features Created with Shape {}".format(df.shape))
        df["DayOfWeek"] = df[self.time_key].dt.day_name()

        if self.expand_days:
            DoW = pd.get_dummies(df.DayOfWeek)
            df = pd.concat([df, DoW], axis=1)
            df["Weekend"] = np.where((df.Saturday == 1) | (df.Sunday == 1), 1, 0)
            df.drop(["DayOfWeek"], axis=1, inplace=True)
        else:
            df["Weekend"] = df["DayOfWeek"].apply(lambda x: 1 if x in ["Saturday", "Sunday"] else 0)

        self.data = df
        self.features = self.data.columns.tolist()
        if self.monotone_constraints:
            for feat in self.features:
                if feat not in [self.key, self.time_key]:
                    self.constraints[feat] = 0
        return self

    def transform(self, X, flag="predict", start_date=None, end_date=None):
        """
        Transforming

        :param X: Pandas DataFrame
        """
        if self.time_key not in X.columns:
            raise ValueError("{} not found in input data. Columns {}".format(self.time_key, X.columns))

        data = self.data.copy()
        if start_date is not None:
            data = data[(data[self.time_key] >= start_date) & (data[self.time_key] <= end_date)]

        X = pd.merge(X, data, on=self.time_key, how="left")
        print("Added Features : ", [feat for feat in self.features if feat != self.time_key])
        print("Post Transforming with Calendar Features data shape is:", X.shape)

        return X
