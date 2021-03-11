import numpy as np
import pandas as pd


# Regressoin Summry generating class
class SummarizeRegression(object):
    """
    Regression Summary Class
        :param date_key: Date column
        :param key: Group / ID columns
    """

    def __init__(self, date_key, key):
        self.date_key = date_key
        self.key = key

    def _deviation_measures(self, y_act, y_fit):

        error = y_act - y_fit
        y_act_mean = np.mean(y_act)
        y_act_var = np.var(y_act, ddof=1)
        n = len(y_act)
        me = np.mean(error)
        if y_act_mean != 0:
            bias = np.mean(y_fit) / np.mean(y_act)
        else:
            bias = np.mean(y_fit)
        max_error = np.max(np.abs(error))
        mae = np.mean(np.abs(error)) / n
        mad = np.mean(np.abs(error - me))
        # epsilon = np.finfo(np.float64).eps
        y_mask = y_act != 0
        # masked_y = y_act
        # masked_y[y_mask] =  1e-10
        mape = np.mean(np.abs(error[y_mask] / y_act[y_mask])) * 100
        rmse = np.sqrt(np.average(error ** 2))

        if y_act_var != 0:
            r2_val = 1.0 - (np.sum(error ** 2) / ((n - 1.0) * y_act_var))
        else:
            r2_val = -np.inf

        # explained varianace
        try:
            numerator = np.average((error - me) ** 2)
            y_true_avg = np.average(y_act)
            denominator = np.average((y_act - y_true_avg) ** 2)
            nonzero_numerator = numerator != 0
            nonzero_denominator = denominator != 0
            valid_score = nonzero_numerator & nonzero_denominator
            output_scores = np.ones(y_act.shape[0])

            output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
            output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
            explained_variance = np.mean(output_scores)
        except Exception as e:
            print(e)
            explained_variance = -1.0
        return me, bias, mae, mad, mape, rmse, r2_val, max_error, explained_variance

    def _print_results(self, values, name=None):
        me, bias, mae, mad, mape, rmse, r2_val, max_error, explained_variance = values

        s = "*" * 80 + "\n"
        s += " Model Summary Statistics"
        s += " - " + name + "\n" if name is not None else "\n"
        s += "*" * 80 + "\n"
        s += "Mean Error (ME)                  :  {:5.4f} \n".format(me)
        s += "Multiplicative Bias              :  {:5.4f} \n".format(bias)
        s += "Mean Abs Error (MAE)             :  {:5.4f} \n".format(mae)
        s += "Mean Abs Deviance Error (MAD)    :  {:5.4f} \n".format(mad)
        s += "Mean Abs Percentage Error(MAPE)  :  {:5.4f} \n".format(mape)
        s += "Root Mean Squared Error (RMSE)   :  {:5.4f} \n".format(rmse)
        s += "R-Squared                        :  {:5.4f} \n".format(r2_val)
        s += "Max Error                        :  {:5.4f} \n".format(max_error)
        s += "Explained Variance               :  {:5.4f} \n".format(explained_variance)
        s += "*" * 80 + "\n"
        print(s)

    def measure(self, y_act, y_fit, name=None):
        results = self._deviation_measures(y_act, y_fit)
        self._print_results(results, name)

    def summarize_date(self):
        values = self.data.groupby(self.date_key).agg({"actual": "sum", "predicted": "sum"})
        results = self._deviation_measures(values["actual"], values["predicted"])
        self._print_results(results, f"At '{self.date_key}' Level")

    def summarize_key(self):
        results = []
        keys = []
        for key_val in self.data[self.key].unique():
            values = self.data[self.data[self.key].isin([key_val])]
            res = self._deviation_measures(values["actual"], values["predicted"])
            results.append(res)
            keys.append(key_val)

        return pd.DataFrame(
            data=results,
            index=keys,
            columns=[
                "ME",
                "Multiplicative Bias",
                "MAE",
                "MAD",
                "MAPE",
                "RMSE",
                "R-squared",
                "Max Error",
                "Explained Variance",
            ],
        )

    def summarize_overall(self):
        results = self._deviation_measures(self.data["actual"], self.data["predicted"])
        self._print_results(results, "At Overall Level")

    def summarize(self, data, actual_key=None, predicted_key=None):
        """
        Summarizing

        :param data: Pandas DataFrame with keys and predictions
        """
        if actual_key is not None:
            self.data = pd.DataFrame(
                {
                    self.date_key: data[self.date_key].values.ravel(),
                    self.key: data[self.key].values.ravel(),
                    "actual": data[actual_key].values.ravel(),
                    "predicted": data[predicted_key].values.ravel(),
                }
            )
        else:
            self.data = data
        self.summarize_overall()
        self.summarize_date()
        df = self.summarize_key()
        print(df)
        return df
