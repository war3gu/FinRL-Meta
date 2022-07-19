import os
import datetime
import urllib3
from dateutil.parser import parse
import threading
import pandas as pd

import tushare as ts

id_market   = 'SSE'
FIELD_DATE = 'trade_date'
data_folder = './data'
token_TS = '51295be6098fe565f6f727019e280ba4821ad5554b551c311bc33ae3'

ts.set_token(token_TS)
pro = ts.pro_api()
csi = pro.index_basic(market=id_market)


class stock_reader():
    def __init__(self):
        print("stockReader")

    def build_url(self, symbol):
        url = symbol
        return url


def download(i, symbol, url, output):
    df1 = ts.pro_bar(ts_code=url, adj='qfq', start_date="19890101", end_date="20020101")
    df2 = ts.pro_bar(ts_code=url, adj='qfq', start_date="20020101", end_date="20211111")

    # df1 = pro.index_daily(ts_code=url, start_date='19890101', end_date='20020101')   #取指数接口，没权限
    # df2 = pro.index_daily(ts_code=url, start_date='20020101', end_date='20211111')

    df = df2

    if df1 is not None:
        df = df1.append(df2)

    df.sort_values(by=FIELD_DATE, ascending=False, inplace=True)  # inplace is important

    df = df.reset_index(drop=True)

    print(df)
    fullPath = os.path.join(output, symbol)
    df.to_csv('{}.csv'.format(fullPath))
    print('download')


def download_all():
    reader = stock_reader()

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    for i, symbol in enumerate(stocks_list):
        url = reader.build_url(symbol)
        download(i, symbol, url, data_folder)


def download_stocks_list():
    data = pro.query('stock_basic',
                     exchange='',
                     list_status='L',
                     fields='ts_code,symbol,name,area,market,industry,list_date')
    print(data)
    data.to_csv('stocks_list.csv')


def download_daily(startIndex, endIndex):
    reader = stock_reader()

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    sl = pd.read_csv('stocks_list.csv',
                     header=0)
    for i in range(startIndex, endIndex):
        df = sl.iloc[i]
        #print(df.ts_code)
        symbol = df.ts_code.lower()
        print(symbol)
        url = reader.build_url(symbol)
        download(i, symbol, url, data_folder)

def translate_to_stock_week(fileName):          #把数据转换一下保存
    fullName = 'datasets/' + fileName
    stock = pd.read_csv(fullName)
    #根据周几进行分割，一周交易五天，不满5天的丢弃。同时把周一的上一个交易日也记录下来，方便数据处理

    df_group = pd.DataFrame()            #还是用dataframe，然后把数据保存起来，每次直接载入，快一些，否则太慢了


    item_last = None
    len = stock.shape[0]
    stock['trade_date'] = stock['trade_date'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m%d'))


    for index in reversed(range(len)):
        item = stock.iloc[index]
        trade_date = item.trade_date
        #trade_date = datetime.datetime.strptime(trade_date, '%Y%m%d')
        weekday = trade_date.weekday()                         #0 周一  6周日

        #判断后面几天是不是一周的几天，是就把包括item_last的一周数据打包保存
        if weekday == 0:
            if index >= 4:
                next1 = stock.iloc[index - 1]
                next2 = stock.iloc[index - 2]
                next3 = stock.iloc[index - 3]
                next4 = stock.iloc[index - 4]

                if (next1.trade_date - item.trade_date).days == 1and \
                        (next2.trade_date - item.trade_date).days == 2and \
                        (next3.trade_date - item.trade_date).days == 3and \
                        (next4.trade_date - item.trade_date).days == 4:
                    df_one = pd.DataFrame()
                    df_one = df_one.append(item_last)
                    df_one = df_one.append(item)
                    df_one = df_one.append(next1)
                    df_one = df_one.append(next2)
                    df_one = df_one.append(next3)
                    df_one = df_one.append(next4)
                    df_group = df_group.append(df_one)

        item_last = item

    outName = 'datasets/' + 'week_' + fileName

    df_group = df_group.loc[:, ~df_group.columns.str.contains('^Unnamed')]

    df_group.to_csv(outName, index=False)


from scipy.stats import zscore
def translate_to_stock_norm_week(fileName):
    fullName = 'datasets/' + fileName
    stock_week = pd.read_csv(fullName)

    begin = 0
    end = stock_week.shape[0]

    df_list = []

    for index in range(begin, end, 6):
        #print(index)
        df_one = pd.DataFrame()
        df_one = df_one.append(stock_week.iloc[index])
        df_one = df_one.append(stock_week.iloc[index+1])
        df_one = df_one.append(stock_week.iloc[index+2])
        df_one = df_one.append(stock_week.iloc[index+3])
        df_one = df_one.append(stock_week.iloc[index+4])
        df_one = df_one.append(stock_week.iloc[index+5])
        df_list.append(df_one)

    df_save = pd.DataFrame()
    #开始归一化，并把周五拿掉
    for df in df_list:
        df_start = df.iloc[0]
        df_5     = df.iloc[1:6]

        df_5['open']      = df_5['open'].apply(lambda x: x/df_start['open'])
        df_5['high']      = df_5['high'].apply(lambda x: x/df_start['high'])
        df_5['low']       = df_5['low'].apply(lambda x: x/df_start['low'])
        df_5['close']     = df_5['close'].apply(lambda x: x/df_start['close'])
        df_5['pre_close'] = df_5['pre_close'].apply(lambda x: x/df_start['pre_close'])
        df_5['change']    = df_5['change'].apply(lambda x: x/df_start['change'])
        df_5['pct_chg']   = df_5['pct_chg'].apply(lambda x: x/df_start['pct_chg'])
        df_5['vol']       = df_5['vol'].apply(lambda x: x/df_start['vol'])
        df_5['amount']    = df_5['amount'].apply(lambda x: x/df_start['amount'])

        df_save = df_save.append(df_5)#把这个整体保存进文件，为了节省时间

    #加一列z-score归一化的close数据作为状态,现有状态太相似
    df_save['z_close'] = zscore(df_save['close'])

    outName = 'datasets/' + 'norm_' + fileName

    df_save.to_csv(outName, index=False)


if __name__ == '__main__':
    #download_daily(0, 10)
    # download_stocks_list()
    # download_all()
    #translate_to_stock_week('000001.sz.csv')
    translate_to_stock_norm_week('week_000001.sz.csv')
