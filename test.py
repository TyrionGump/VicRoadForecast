import numpy as np
import itertools
import pandas as pd

from config import settings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import MSELoss
from torch.nn import L1Loss

import torch
from torch import nn
from d2l import torch as d2l


a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 2, 3], [4, 5, 6]])
print(np.concatenate([a, b], axis=0).reshape(len([a, b]), len(a), len(a[0])))

enc_dec = d2l.EncoderDecoder()