# -*- coding: utf-8 -*-
"""
@File:link_gdf.geojson-PyCharm-data_to_tensor.py
@Date: 14/10/21
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

import torch
from torch.utils.data import Dataset, DataLoader


class TorchDataset(Dataset):
    def __init__(self, feature_set, target_set):
        #定义好 image 的路径
        self.features = torch.from_numpy(feature_set).float()
        self.targets = torch.from_numpy(target_set).float()

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def __len__(self):
        return len(self.features)




