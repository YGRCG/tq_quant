#!/usr/bin/env python
#  -*- coding: utf-8 -*-
__author__ = 'chengzhi'

from datetime import datetime
from contextlib import closing
from tqsdk import TqApi, TqAuth
from tqsdk.tools import DataDownloader
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..',  '.env'))
ACCOUNT = os.getenv('ACCOUNT')
PASSWORD = os.getenv('PASSWORD')

api = TqApi(auth=TqAuth(ACCOUNT, PASSWORD))
# 下载从 2018-01-01凌晨6点 到 2018-06-01下午4点 的 cu1805 分钟线数据
# kd = DataDownloader(api, symbol_list="SHFE.cu1805", dur_sec=60,
#                     start_dt=datetime(2018, 1, 1, 6, 0 ,0), end_dt=datetime(2018, 6, 1, 16, 0, 0), csv_file_name="kline.csv")
# 下载从 2018-05-01凌晨0点 到 2018-07-01凌晨0点 的 T1809 盘口Tick数据
td = DataDownloader(api, symbol_list="SHFE.fu2609", dur_sec=0,
                    start_dt=datetime(2026, 4, 21), end_dt=datetime(2026, 5, 22), csv_file_name="../data/fu2609/tick.csv")
# 使用with closing机制确保下载完成后释放对应的资源
with closing(api):
    while not kd.is_finished() or not td.is_finished():
        api.wait_update()
        print("progress: kline: %.2f%% tick:%.2f%%" % (kd.get_progress(), td.get_progress()))