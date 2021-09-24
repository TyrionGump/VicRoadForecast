"""
@file:VicRoadForecast-PyCharm-forecaster.py
@time: 23/9/21
@author: Yubo Sun
@e-mail: tyriongump@gmail.com
@github: TyrionGump
@Team: TrafficO Developers
@copyright: The University of Melbourne
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d - %(module)s')


class SeparateModel:
    def __init__(self, features_dict, target_dict):
        self.features_dict = features_dict
        self.target_dict = target_dict
        self.X_train_dict = {}
        self.y_train_dict = {}
        self.X_test_dict = {}
        self.y_test_dict = {}
        self.y_pred_dict = {}

        self.models = {}
        self.scalar = StandardScaler()
        self.mse_dict = {}
        self.mape_dict = {}

        self.logger = logging.getLogger(__name__)

    def train_models(self):
        self._train_test_split_uniform()
        self._train_models()

    def _train_test_split_uniform(self):
        self.logger.info('Splitting training set and testing set...')
        for link_id in self.features_dict:
            X_train, X_test, y_train, y_test = train_test_split(self.features_dict[link_id],
                                                                self.target_dict[link_id],
                                                                test_size=0.2,
                                                                random_state=42)

            self.scalar.fit(X_train)
            self.X_train_dict[link_id] = self.scalar.transform(X_train)
            self.X_test_dict[link_id] = self.scalar.transform(X_test)
            self.y_train_dict[link_id] = y_train
            self.y_test_dict[link_id] = y_test

    def _train_models(self):
        self.logger.info('Training models...')
        for link_id in self.X_train_dict:
            m = LinearRegression()
            m.fit(self.X_train_dict[link_id], self.y_train_dict[link_id])
            self.y_pred_dict[link_id] = m.predict(self.X_test_dict[link_id])
            joblib.dump(m, '../models/{}_lr.model'.format(link_id))

    def cal_mse(self):
        self.logger.info('Evaluating models based on mean square error...')
        avg_mse = 0
        for link_id in self.y_test_dict:
            self.mse_dict[link_id] = mean_squared_error(y_true=self.y_test_dict[link_id],
                                                        y_pred=self.y_pred_dict[link_id])
            avg_mse += self.mse_dict[link_id]
        avg_mse /= len(self.y_test_dict)
        print('Average MSE: ', avg_mse)

    def cal_mape(self):
        self.logger.info('Evaluating models based on mean absolute perentage error...')
        avg_mape = 0
        for link_id in self.y_test_dict:
            self.mape_dict[link_id] = mean_absolute_percentage_error(y_true=self.y_test_dict[link_id],
                                                                     y_pred=self.y_pred_dict[link_id])
            print(self.mape_dict[link_id])
            avg_mape += self.mape_dict[link_id]
        avg_mape /= len(self.y_test_dict)
        print('Average MAPE: ', avg_mape)














