"""
@file:VicRoadForecast-PyCharm-forecaster.py
@time: 23/9/21
@author: Yubo Sun
@e-mail: tyriongump@gmail.com
@github: TyrionGump
@Team: TrafficO Developers
@copyright: The University of Melbourne
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d - %(module)s')


class SeparateModel:
    def __init__(self, training_feature_dict, training_target_dict, testing_feature_dict, testing_target_dict,
                 seed=42, validation_size=0.2, model_type=LinearRegression):
        self.training_feature_dict = training_feature_dict
        self.training_target_dict = training_target_dict
        self.testing_feature_dict = testing_feature_dict
        self.testing_target_dict = testing_target_dict
        self.seed = seed
        self.validation_size = validation_size
        self.model_type = model_type
        self.link_ids = list(training_feature_dict.keys())

        self.x_training_dict = {}
        self.y_training_dict = {}
        self.x_validation_dict = {}
        self.y_validation_dict = {}

        self.y_validation_pred_dict = {}
        self.y_testing_pred_dict = {}

        self.validation_mse_dict = {}
        self.validation_mape_dict = {}
        self.testing_mse_dict = {}
        self.testing_mape_dict = {}

        self.avg_validation_mse = None
        self.avg_validation_mape = None
        self.avg_testing_mse = None
        self.avg_testing_mape = None

        self.scalars = {}
        self.models = {}

        self.logger = None
        self._set_logger()

    def train_models(self, save):
        self._split_training_validation()
        self._normalize(x_training_set=self.x_training_dict, x_testing_set=self.x_validation_dict)
        self._fit(x_training_set=self.x_training_dict, y_training_set=self.y_training_dict)
        self.y_validation_pred_dict = self._pred(x_testing_set=self.x_validation_dict)

        self._normalize(x_training_set=self.training_feature_dict, x_testing_set=self.testing_feature_dict)
        self._fit(x_training_set=self.training_feature_dict, y_training_set=self.training_target_dict)
        self.y_testing_pred_dict = self._pred(x_testing_set=self.testing_feature_dict)

        if save:
            self._save_model()
            self._save_prediction_result()

    def get_validation_true_pred(self):
        return self.y_validation_dict, self.y_validation_pred_dict

    def get_testing_true_pred(self):
        return self.testing_target_dict, self.y_testing_pred_dict

    def _split_training_validation(self):
        for link_id in self.link_ids:
            x_train, x_test, y_train, y_test = train_test_split(
                self.training_feature_dict[link_id],
                self.training_target_dict[link_id],
                test_size=self.validation_size,
                random_state=self.seed)

            self.x_training_dict[link_id] = x_train.reset_index(drop=True)
            self.x_validation_dict[link_id] = x_test.reset_index(drop=True)
            self.y_training_dict[link_id] = y_train.reset_index(drop=True)
            self.y_validation_dict[link_id] = y_test.reset_index(drop=True)

    def _normalize(self, x_training_set, x_testing_set):
        for link_id in self.link_ids:
            scalar = StandardScaler()
            scalar.fit(x_training_set[link_id])
            x_training_set[link_id] = scalar.transform(x_training_set[link_id])
            x_testing_set[link_id] = scalar.transform(x_testing_set[link_id])
            self.scalars[link_id] = scalar

    def _fit(self, x_training_set, y_training_set):
        for link_id in self.link_ids:
            m = self.model_type()
            m.fit(x_training_set[link_id], y_training_set[link_id])
            self.models[link_id] = m

    def _pred(self, x_testing_set):
        pred = {}
        for link_id in self.link_ids:
            pred[link_id] = self.models[link_id].predict(x_testing_set[link_id])
        return pred

    def _set_logger(self):
        self.logger = logging.getLogger('forecaster')
        format_str = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.logger.setLevel(logging.INFO)
        th = logging.FileHandler(filename='../logs/forecaster.log', mode='w', encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(th)

    def _save_model(self):
        model_name = self.model_type().__str__()[:-2]
        os.makedirs('../models/{}'.format(model_name), exist_ok=True)
        for k, v in self.models.items():
            joblib.dump(v, '../models/{}/{}.model'.format(model_name, k))

    def _save_prediction_result(self):
        model_name = self.model_type().__str__()[:-2]
        os.makedirs('../result/{}/prediction/validation_set'.format(model_name), exist_ok=True)
        os.makedirs('../result/{}/prediction/testing_set'.format(model_name), exist_ok=True)

        for link_id in self.link_ids:
            columns = ['pred_' + i for i in self.y_validation_dict[link_id].columns]
            df = pd.DataFrame(self.y_validation_pred_dict[link_id], columns=columns)
            pd.concat([self.y_validation_dict[link_id], df], axis=1).to_csv(
                '../result/{}/prediction/validation_set/{}.csv'.format(model_name, link_id), index=False)


            columns = ['pred_' + i for i in self.testing_target_dict[link_id].columns]
            df = pd.DataFrame(self.y_testing_pred_dict[link_id], columns=columns)
            pd.concat([self.testing_target_dict[link_id], df], axis=1).to_csv(
                '../result/{}/prediction/testing_set/{}.csv'.format(model_name, link_id), index=False)



















