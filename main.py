# -*- coding: utf-8 -*-
"""
@File:main.py
@Date: 2021/11/25
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

from forecast_system.control_center import ControlCenter

if __name__ == '__main__':
    sys = ControlCenter()
    # sys.pull_traffic_data()
    # sys.add_features()
    # sys.create_dataset()
    sys.train()