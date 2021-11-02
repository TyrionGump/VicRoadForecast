# -*- coding: utf-8 -*-
"""
@File:link_gdf.geojson-PyCharm-evaluator.py
@Date: 15/10/21
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import os


class Evaluator:
    """

    This class receives the prediction results and the true values of each link. Multiple metrics will be calculated
    for each link to represent the prediction performance of the model.

    """
    def __init__(self, validation_true, validation_pred, testing_true, testing_pred, model_type):
        """Constructor of the evaluator

        Args:
            validation_true: A dictionary contains the true values of the validation set of each link
            validation_pred: A dictionary contains the prediction results of the validation set of each link
            testing_true: A dictionary contains the true values of the testing set of each link
            testing_pred: A dictionary contains the prediction results of the testing set of each link
            model_type: Class of sklearn
        """
        self.validation_true = validation_true
        self.validation_pred = validation_pred
        self.testing_true = testing_true
        self.testing_pred = testing_pred
        self.model_type = model_type

        self.link_ids = validation_pred.keys()
        self.performance_df = {'ID': self.link_ids}

    def evaluate(self, save):
        """Calculate metrics for the model

        Calculate MSE, MAPE, MAE on the validation set and testing set. Then, store the evaluation results in
        the local client.

        Args:
            save: Boolean flag represents whether saving evaluation results in the local client.
        """
        self.performance_df['Validation_MSE'] = self._cal_mse(self.validation_true, self.validation_pred)
        self.performance_df['Testing_MSE'] = self._cal_mse(self.testing_true, self.testing_pred)

        self.performance_df['Validation_MAPE'] = self._cal_mape(self.validation_true, self.validation_pred)
        self.performance_df['Testing_MAPE'] = self._cal_mape(self.testing_true, self.testing_pred)

        self.performance_df['Validation_MAE'] = self._cal_mae(self.validation_true, self.validation_pred)
        self.performance_df['Testing_MAE'] = self._cal_mae(self.testing_true, self.testing_pred)

        self.performance_df = pd.DataFrame.from_dict(self.performance_df)

        if save:
            model_name = self.model_type().__str__()[:-2]
            os.makedirs('../result/{}/performance'.format(model_name), exist_ok=True)
            self.performance_df.to_csv('../result/{}/performance/accuracy.csv'.format(model_name), index=False)

    def _cal_mse(self, true_dict, pred_dict):
        mse_ls = []
        for link_id in self.link_ids:
            mse_ls.append(mean_squared_error(y_true=true_dict[link_id], y_pred=pred_dict[link_id]))
        return mse_ls

    def _cal_mape(self, true_dict, pred_dict):
        mape_ls = []
        for link_id in self.link_ids:
            mape_ls.append(mean_absolute_percentage_error(y_true=true_dict[link_id], y_pred=pred_dict[link_id]))
        return mape_ls

    def _cal_mae(self, true_dict, pred_dict):
        mae_ls = []
        for link_id in self.link_ids:
            mae_ls.append(mean_absolute_error(y_true=true_dict[link_id], y_pred=pred_dict[link_id]))
        return mae_ls







