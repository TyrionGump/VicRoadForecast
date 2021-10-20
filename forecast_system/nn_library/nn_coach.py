# -*- coding: utf-8 -*-
"""
@File:link_gdf.geojson-PyCharm-nn_coach.py
@Date: 14/10/21
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""
from forecast_system.nn_library.data_to_tensor import TorchDataset
from forecast_system.nn_library.rnn import RNN
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from tqdm import tqdm

training_df_2727_feature = pd.read_csv('../../data/MELBOURNE CITY/1007110000-1014110000/feature/1374.csv')
training_df_2727_target = pd.read_csv('../../data/MELBOURNE CITY/1007110000-1014110000/target/1374.csv')

testing_df_2727_feature = pd.read_csv('../../data/MELBOURNE CITY/1014110000-1015110000/feature/1374.csv')
testing_df_2727_target = pd.read_csv('../../data/MELBOURNE CITY/1014110000-1015110000/target/1374.csv')

scaler = StandardScaler()
training_feature_set = scaler.fit_transform(training_df_2727_feature.values)
training_target_set = training_df_2727_target.values
testing_feature_set = scaler.transform(testing_df_2727_feature.values)
testing_target_set = testing_df_2727_target.values

feature_size = 2
target_size = 10
feature_time_steps = 20
hidden_size = 5
num_layers = 1
batch_size = 500


training_set = TorchDataset(training_feature_set, training_target_set)
training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
testing_set = TorchDataset(testing_feature_set, testing_target_set)
testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=False)


model = RNN(feature_size=feature_size, target_size=target_size, hidden_size=hidden_size, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(1000):
    loss_track = 0
    for feature, target in training_loader:
        hidden_prev = torch.zeros(num_layers, feature.shape[0], hidden_size)
        x = feature.view(feature.shape[0], feature_time_steps, feature_size)
        output, _ = model(x, hidden_prev)
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        loss_track += loss.item() * feature.shape[0]
    loss_track /= len(training_set)
    if epoch % 100 == 0:
        print("iteration: {} loss {}".format(epoch, loss_track))

loss_track = 0
y_pred = []
y_true = []

model.eval()
for feature, target in testing_loader:
    hidden_prev = torch.zeros(num_layers, feature.shape[0], hidden_size)
    x = feature.view(feature.shape[0], feature_time_steps, feature_size)
    output, _ = model(x, hidden_prev)

    y_pred.extend(output.data.numpy().tolist())
    y_true.extend(target.numpy().tolist())
    loss = F.mse_loss(output, target)
    loss_track += loss.item() * feature.shape[0]
loss_track /= len(testing_set)
print("iteration: testing loss {}".format(loss_track))


print(mean_squared_error(y_true=y_true, y_pred=y_pred))
print(mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred))

plt.title('RNN result for Link 1374 (Derrimut Rd) between 2021-10-14 11:00:00 - 2021-10-15 11:00:00', fontsize=8)
plt.plot(list(range(len(y_true))), [i[0] for i in y_true], color='blue', linewidth=0.5, label='true travel time t + 1')
plt.plot(list(range(len(y_pred))), [i[0] for i in y_pred], color='red', linewidth=0.5, label='pred travel time t + 1')
plt.ylabel('travel time')
plt.xlabel('time window (30s)')
plt.legend()
plt.show()

# true_time_series = [i[0] for i in y_true]
#
# plt.figure(figsize=(16, 9))
# plt.plot(list(range(len(true_time_series))), true_time_series, color='blue')
# for i in tqdm(range(len(y_true))):
#     plt.scatter(list(range(i, i + len(y_true[i]))), y_pred[i], s=10)
#
# plt.show()




