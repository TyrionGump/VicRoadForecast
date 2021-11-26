# -*- coding: utf-8 -*-
"""
@File:weather_observers.py
@Date: 2021/11/14
@Author: Yubo Sun
@E-mail: tyriongump@gmail.com
@Github: TyrionGump
@Team: TrafficO Developers
@Copyright: The University of Melbourne
"""

import requests
from datetime import datetime
url = "https://api.ambeedata.com/weather/history/by-lat-lng"
querystring = {"lat":"-37.7705421","lng":"145.079041","from":"2021-11-12 00:00:00","to":"2021-11-15 00:00:00"}
headers = {
    'x-api-key': "8a90d43f683d7ab484d9eeadbfabf0b71556c45aa90a69ba968a6f2894a47d9f",
    'Content-type': "application/json"
    }
response = requests.request("GET", url, headers=headers, params=querystring)
print(response.text)
print(datetime.fromtimestamp(1636678800))
print(datetime.fromtimestamp(1636930800))