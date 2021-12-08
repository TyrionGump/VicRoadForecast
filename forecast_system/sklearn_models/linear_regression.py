# -*- coding: utf-8 -*-
"""
@File:linear_regression.py
@Date: 2021/12/1
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from config import args
import os
import logging
import joblib


class LR_COACH:
    def __init__(self, training_set: dict, testing_set: dict, normalize_encoder):
        self.link_ids = training_set['feature'].keys()
        self.training_set = training_set
        self.testing_set = testing_set
        self.normalize_encoder = normalize_encoder

    def run(self):
        if args.separate:
            os.makedirs('result/linear_regression/separate/models', exist_ok=True)
            os.makedirs('result/linear_regression/separate/prediction_values', exist_ok=True)
            os.makedirs('result/linear_regression/separate/true_values', exist_ok=True)
            os.makedirs('result/linear_regression/separate/performance_results', exist_ok=True)

            for link_id in self.link_ids:
                logging.info('Running Linear Regression for link {}'.format(link_id))
                model = LinearRegression()
                model.fit(self.training_set['feature'][link_id], self.training_set['target'][link_id])
                joblib.dump(model, 'result/linear_regression/separate/models/{}.model'.format(link_id))

                y_pred = model.predict(self.testing_set['feature'][link_id])
                scale_data_max = self.normalize_encoder[link_id].data_max_[0]
                scale_data_min = self.normalize_encoder[link_id].data_min_[0]
                y_pred = y_pred * (scale_data_max - scale_data_min) + scale_data_min
                y_true = self.testing_set['target'][link_id] * (scale_data_max - scale_data_min) + scale_data_min
                performance = self.eval(y_true, y_pred)

                y_pred = pd.DataFrame(y_pred)
                y_true = pd.DataFrame(y_true)
                y_pred.to_csv('result/linear_regression/separate/prediction_values/{}.csv'.format(link_id), index=False)
                y_true.to_csv('result/linear_regression/separate/true_values/{}.csv'.format(link_id), index=False)
                performance.to_csv('result/linear_regression/separate/performance_results/{}.csv'.format(link_id), index=False)

        else:
            os.makedirs('result/linear_regression/global/models', exist_ok=True)
            os.makedirs('result/linear_regression/global/prediction_values', exist_ok=True)
            os.makedirs('result/linear_regression/global/true_values', exist_ok=True)
            os.makedirs('result/linear_regression/global/performance_results', exist_ok=True)

            training_feature_all = [self.training_set['feature'][link_id] for link_id in self.link_ids]
            training_target_all = [self.training_set['target'][link_id] for link_id in self.link_ids]

            model = LinearRegression()
            model.fit(np.concatenate(training_feature_all, axis=0), np.concatenate(training_target_all, axis=0))

            joblib.dump(model, 'result/linear_regression/global/models/global.model')
            for link_id in self.link_ids:
                y_pred = model.predict(self.testing_set['feature'][link_id])
                scale_data_max = self.normalize_encoder.data_max_[0]
                scale_data_min = self.normalize_encoder.data_min_[0]
                y_pred = y_pred * (scale_data_max - scale_data_min) + scale_data_min
                y_true = self.testing_set['target'][link_id] * (scale_data_max - scale_data_min) + scale_data_min
                performance = self.eval(y_true, y_pred)

                y_pred = pd.DataFrame(y_pred)
                y_true = pd.DataFrame(y_true)
                y_pred.to_csv('result/linear_regression/global/prediction_values/{}.csv'.format(link_id), index=False)
                y_true.to_csv('result/linear_regression/global/true_values/{}.csv'.format(link_id), index=False)
                performance.to_csv('result/linear_regression/global/performance_results/{}.csv'.format(link_id),
                                   index=False)

    @staticmethod
    def eval(y_true, y_pred):
        performance_df = pd.DataFrame()
        performance_df['Step'] = ['t+{}'.format(i+1) for i in range(y_pred.shape[1])]
        performance_df['MSE'] = mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
        performance_df['MAE'] = mean_absolute_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
        performance_df['MAPE'] = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
        return performance_df



