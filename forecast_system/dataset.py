from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import math
import pandas as pd
import numpy as np
import copy
import logging
from config import args
from forecast_system.nn_library.nn_utils import TorchDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d - %(module)s')


class Dataset:
    def __init__(self, training_data_dict, testing_data_dict, separate=False):
        self.training_data_dict = copy.deepcopy(training_data_dict)
        self.training_feature_dict = {}
        self.training_target_dict = {}

        self.testing_data_dict = copy.deepcopy(testing_data_dict)
        self.testing_feature_dict = {}
        self.testing_target_dict = {}

        self.link_ids = self.training_data_dict.keys()
        self.normalize_encoder = {}

        self.separate = separate
        self.logger = None
        self._set_logger()
        self._drop_id_timestamp()

    def _drop_id_timestamp(self):
        for link_id in self.link_ids:
            self.training_data_dict[link_id].drop(columns=['ID', 'timestamp'], inplace=True)
            self.testing_data_dict[link_id].drop(columns=['ID', 'timestamp'], inplace=True)

    def get_training_set(self):
        return {'feature': self.training_feature_dict, 'target': self.training_target_dict}

    def get_testing_set(self):
        return {'feature': self.testing_feature_dict, 'target': self.testing_target_dict}

    def get_normalize_encoder(self):
        return self.normalize_encoder

    def tt_to_speed(self):
        for link_id in self.link_ids:
            self.training_data_dict[link_id]['travel_time'] = self.training_data_dict[link_id]['length'] / self.training_data_dict[link_id]['travel_time']
            self.testing_data_dict[link_id]['travel_time'] = self.testing_data_dict[link_id]['length'] / self.testing_data_dict[link_id]['travel_time']

    def aggregate_time_series(self, minute=5):
        for link_id in self.link_ids:
            columns = self.training_data_dict[link_id].columns
            aggregate_training_data = []
            aggregate_testing_data = []

            t = 0
            while (t + 1) * minute * 2 - 1 <= len(self.training_data_dict[link_id]):
                aggregate_training_data.append(
                    self.training_data_dict[link_id].loc[t * minute * 2: (t + 1) * minute * 2 - 1, :].mean().to_list())
                t += 1

            t = 0
            while (t + 1) * minute * 2 - 1 <= len(self.testing_data_dict[link_id]):
                aggregate_testing_data.append(
                    self.testing_data_dict[link_id].loc[t * minute * 2: (t + 1) * minute * 2 - 1, :].mean().to_list())
                t += 1

            self.training_data_dict[link_id] = pd.DataFrame(aggregate_training_data, columns=columns)
            self.testing_data_dict[link_id] = pd.DataFrame(aggregate_testing_data, columns=columns)
        # print(self.training_data_dict[162])

    def one_hot_encode(self):
        """

        There are some disputes that hour of day and day of week belongs to categorical variables or numeric variables.
        Also, some people said we can transform these features into radians of a circle.

        """
        pass

    def normalize(self):
        if self.separate:
            for link_id in self.link_ids:
                columns = self.training_data_dict[link_id].columns
                encoder = MinMaxScaler()
                normalized_training_data = encoder.fit_transform(self.training_data_dict[link_id])
                normalized_testing_data = encoder.transform(self.testing_data_dict[link_id])
                self.training_data_dict[link_id] = pd.DataFrame(normalized_training_data, columns=columns)
                self.testing_data_dict[link_id] = pd.DataFrame(normalized_testing_data, columns=columns)
                self.normalize_encoder[link_id] = encoder
        else:
            all_in_one_training = [self.training_data_dict[link_id] for link_id in self.link_ids]
            all_in_one_training = pd.concat(all_in_one_training, axis=0)
            self.normalize_encoder = MinMaxScaler()
            self.normalize_encoder.fit(all_in_one_training)
            for link_id in self.link_ids:
                columns = self.training_data_dict[link_id].columns
                normalized_training_data = self.normalize_encoder.transform(self.training_data_dict[link_id])
                normalized_testing_data = self.normalize_encoder.transform(self.testing_data_dict[link_id])
                self.training_data_dict[link_id] = pd.DataFrame(normalized_training_data, columns=columns)
                self.testing_data_dict[link_id] = pd.DataFrame(normalized_testing_data, columns=columns)

    def inverse_normalize(self):
        for link_id in self.link_ids:
            columns = self.training_data_dict[link_id].columns
            if self.separate:
                encoder = self.normalize_encoder[link_id]
            else:
                encoder = self.normalize_encoder
            inverse_training_data = encoder.inverser_transform(self.training_data_dict[link_id])
            inverse_testing_data = encoder.inverser_transform(self.testing_data_dict[link_id])

            self.training_data_dict[link_id] = pd.DataFrame(inverse_training_data, columns=columns)
            self.testing_data_dict[link_id] = pd.DataFrame(inverse_testing_data, columns=columns)

    def _set_logger(self):
        self.logger = logging.getLogger('data_harvester')
        format_str = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.logger.setLevel(logging.INFO)
        # sh = logging.StreamHandler()
        # sh.setFormatter(format_str)
        th = logging.FileHandler(filename=args.file_config['log_file']['DATASET'], mode='w', encoding='utf-8')
        th.setFormatter(format_str)
        # self.logger.addHandler(sh)
        self.logger.addHandler(th)


class RNN_Dataset(Dataset):
    def __init__(self, training_data_dict, testing_data_dict, separate=False):
        super().__init__(training_data_dict, testing_data_dict, separate)
        self.training_tensor_set = None
        self.testing_tensor_set = None

    def reshape(self, forward_steps=36, backward_steps=12):
        for link_id in self.link_ids:
            self.training_feature_dict[link_id], self.training_target_dict[link_id] = self._rolling_time_series(
                self.training_data_dict[link_id], forward_steps, backward_steps)
            self.testing_feature_dict[link_id], self.testing_target_dict[link_id] = self._rolling_time_series(
                self.testing_data_dict[link_id], forward_steps, backward_steps)

    def _rolling_time_series(self, df: pd.DataFrame, forward_steps, backward_steps):
        window_size = forward_steps + backward_steps
        rolled_df = df.copy()
        rolled_df = rolled_df.rolling(window=window_size, center=False)
        feature = []
        target = []
        for df_subset in rolled_df:
            if len(df_subset) < window_size or df_subset.index.to_list()[-1] >= df.shape[0] - 1:
                continue
            feature.append(df_subset.values.tolist()[:forward_steps])
            target.append(df_subset['travel_time'].values.tolist()[forward_steps:])
        feature = np.array(feature)
        target = np.array(target)
        # target = target * (self.normalize_encoder.data_max_[0] - self.normalize_encoder.data_min_[0]) + self.normalize_encoder.data_min_[0]
        return feature, target


class LR_Dataset(Dataset):
    def __init__(self, training_data_dict, testing_data_dict, separate=False):
        super().__init__(training_data_dict, testing_data_dict, separate)
        self.training_set = {}
        self.testing_set = {}

    def reshape(self, forward_steps=36, backward_steps=12):
        for link_id in self.link_ids:
            self.training_feature_dict[link_id], self.training_target_dict[link_id] = self._rolling_time_series(
                self.training_data_dict[link_id], forward_steps, backward_steps)
            self.testing_feature_dict[link_id], self.testing_target_dict[link_id] = self._rolling_time_series(
                self.testing_data_dict[link_id], forward_steps, backward_steps)

    def _rolling_time_series(self, df: pd.DataFrame, forward_steps, backward_steps):
        window_size = forward_steps + backward_steps
        rolled_df = df.copy()
        rolled_df = rolled_df.rolling(window=window_size, center=False)
        feature = []
        target = []
        for df_subset in rolled_df:
            if len(df_subset) < window_size or df_subset.index.to_list()[-1] >= df.shape[0] - 1:
                continue
            feature.append(df_subset.values.tolist()[:forward_steps])
            target.append(df_subset['travel_time'].values.tolist()[forward_steps:])
        feature = np.array(feature)
        feature = feature.reshape((feature.shape[0], feature.shape[1] * feature.shape[2]))
        target = np.array(target)
        # target = target * (self.normalize_encoder.data_max_[0] - self.normalize_encoder.data_min_[0]) + self.normalize_encoder.data_min_[0]
        return feature, target





