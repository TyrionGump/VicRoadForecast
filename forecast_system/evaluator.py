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
import matplotlib.pyplot as plt
import os

class Evaluator:
    def __init__(self, validation_true, validation_pred, testing_true, testing_pred, model_type):
        self.validation_true = validation_true
        self.validation_pred = validation_pred
        self.testing_true = testing_true
        self.testing_pred = testing_pred
        self.model_type = model_type

        self.link_ids = validation_pred.keys()
        self.performance_df = {'ID': self.link_ids}

        y_true = testing_true[1374]
        y_pred = testing_pred[1374]

        plt.title('Linear Regression result for Link 1374 (Derrimut Rd) between 2021-10-14 11:00:00 - 2021-10-15 11:00:00 ', fontsize=8)
        plt.plot(list(range(len(y_true))), [i[0] for i in y_true.values.tolist()], color='blue', linewidth=0.5,
                 label='true travel time t + 1')
        plt.plot(list(range(len(y_pred))), [i[0] for i in y_pred.tolist()], color='red', linewidth=0.5,
                 label='pred travel time t + 1')
        plt.ylabel('travel time')
        plt.xlabel('time window (30s)')
        plt.legend()
        plt.show()


    def evaluate(self, save):
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







