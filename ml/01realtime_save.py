#!/usr/bin/env python
#  -*- coding: utf-8 -*-
__author__ = 'chengzhi'

from tqsdk import TqApi, TqAuth
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..',  '.env'))
ACCOUNT = os.getenv('ACCOUNT')
PASSWORD = os.getenv('PASSWORD')

# 创建API实例,传入自己的快期账户
api = TqApi(auth=TqAuth(ACCOUNT, PASSWORD))
# 获得上期所 fu2609 的行情引用，当行情有变化时 quote 中的字段会对应更新
quote = api.get_quote("SHFE.fu2609")

while True:
    # 调用 wait_update 等待业务信息发生变化，例如: 行情发生变化, 委托单状态变化, 发生成交等等
    # 注意：其他合约的行情的更新也会触发业务信息变化，因此下面使用 is_changing 判断 FG209 的行情是否有变化
    api.wait_update()
    # 如果 fu2609 的任何字段有变化，is_changing就会返回 True
    if api.is_changing(quote):
        print("行情变化", quote)
    # 只有当 fu2609 的最新价有变化，is_changing才会返回 True
    if api.is_changing(quote, "last_price"):
        print("最新价变化", quote.last_price)
    # 当 fu2609 的买1价/买1量/卖1价/卖1量中任何一个有变化，is_changing都会返回 True
    if api.is_changing(quote, ["ask_price1", "ask_volume1", "bid_price1", "bid_volume1"]):
        print("盘口变化", quote.ask_price1, quote.ask_volume1, quote.bid_price1, quote.bid_volume1)