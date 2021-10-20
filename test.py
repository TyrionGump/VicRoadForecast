# -*- coding: utf-8 -*-
"""
@File:VicRoadForecast-PyCharm-test.py
@Date: 30/9/21
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""
import math

import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
import numpy as np

# x = [math.log(i, 10) for i in [10, 100, 1000, 10000, 100000]]
# y = [33, 33, 27, 95, 899]
#
# plt.plot(x, y, marker='*')
# plt.xlabel('log(row_num, 10)')
# plt.ylabel('query time/s')
# plt.show()
#
print(datetime(2021, 10, 7, 0, 0, 0).timestamp())
print(datetime.fromtimestamp(1633651200), datetime.fromtimestamp(1633608000))
# # print(datetime.fromtimestamp(1633348800), datetime.fromtimestamp(1633392000))
# # print(datetime.fromtimestamp(1633392000), datetime.fromtimestamp(1633435200))
# # print(datetime.fromtimestamp(1633435200), datetime.fromtimestamp(1633478400))
# # print(datetime.fromtimestamp(1622851200))
#
# # a = pd.DataFrame([[1, None, 2],
# #                   [3, 2, 4],
# #                   [1, None, 3]])
# # print(a)
# # print(a.interpolate(method='linear', axis=0, limit_direction='both'))
#
# a = [[1, 2, 3], [4, 5, 6]]
# b = [[3, 5, 5], [5, 6, 7]]
# print(mean_squared_error(y_true=b, y_pred=a, multioutput='raw_values'))
# print(mean_squared_error(y_true=b, y_pred=a))

# a = np.array([[1, 2, 3], [4, 5, 6]]).T.reshape(-1)
# print(a.tolist())
# print(a.shape)
#
# print(a.T.reshape(-1))

# a = datetime.now()
# print(a.tzinfo)
#
# from pylab import *
#
# cmap = cm.get_cmap('seismic', 5)    # PiYG
#
# for i in range(cmap.N):
#     rgba = cmap(i)
#     # rgb2hex accepts rgb or rgba
#     print(matplotlib.colors.rgb2hex(rgba))
