#!/usr/bin/env python
# coding: utf-8

# ## Quantitative trading in China A stock market with FinRL

# ### Import modules

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from IPython import display

display.set_matplotlib_formats("svg")

from finrl_meta import config
from finrl_meta.data_processors.processor_tusharepro import TushareProProcessor, ReturnPlotter
from finrl_meta.env_stock_trading.env_stocktrading_A_war3gu import StockTradingEnv
from drl_agents.stablebaselines3_models import DRLAgent

pd.options.display.max_columns = None

import pyfolio
from pyfolio import timeseries

import multiprocessing
import platform

from stable_baselines3 import TD3
from stable_baselines3 import SAC

print("ALL Modules have been imported!")

# ### Create folders

import os

if not os.path.exists("./datasets"):
    os.makedirs("./datasets")
if not os.path.exists("./trained_models"):
    os.makedirs("./trained_models")
if not os.path.exists("./tensorboard_log"):
    os.makedirs("./tensorboard_log")
if not os.path.exists("./results"):
    os.makedirs("./results")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ### Download data, cleaning and feature engineering
'''
ticket_list = ['600000.SH', '600009.SH', '600016.SH', '600028.SH', '600030.SH',
               '600031.SH', '600036.SH', '600050.SH', '600104.SH', '600196.SH',
               '600276.SH', '600309.SH', '600519.SH', '600547.SH', '600570.SH']
'''

ticket_list = ['600000.SH', '600009.SH', '600016.SH']

train_start_date = '2015-01-01'
train_stop_date = '2019-08-01'
val_start_date = '2019-08-01'
val_stop_date = '2021-01-03'

token = '27080ec403c0218f96f388bca1b1d85329d563c91a43672239619ef5'


ts_processor = TushareProProcessor("tusharepro", token=token)
ts_processor.download_data(ticket_list, train_start_date, val_stop_date, "1D")
ts_processor.clean_data()
ts_processor.dataframe

ts_processor.add_technical_indicator(config.TECHNICAL_INDICATORS_LIST)
ts_processor.clean_data()


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    train = ts_processor.data_split(ts_processor.dataframe, train_start_date, train_stop_date)
    trade = ts_processor.data_split(ts_processor.dataframe, val_start_date, val_stop_date)

    print("train length = {0}".format(len(train.date.unique())))
    print("trade length = {0}".format(len(trade.date.unique())))

    print(train.head())
    print(trade.head())

    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension * (len(config.TECHNICAL_INDICATORS_LIST) + 2) + 1
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    total_timesteps = 500000  # 总的采样次数,不能太少。一局1000天，相当于玩了1000局，有点少

    env_kwargs_train = {
        "stock_dim": stock_dimension,
        "hmax": 1000,
        "initial_amount": 1000000,
        "buy_cost_pct": 6.87e-5,
        "sell_cost_pct": 1.0687e-3,
        "reward_scaling": 1e-1,
        "state_space": state_space,
        "action_space": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "print_verbosity": 100,                     #多少个episode结束打一次log
        "initial_buy": True,
        "hundred_each_trade": True,
        "out_of_cash_penalty": 0.001,
        "cash_limit": 0.2,
        "model_name":"stock_a",
        "mode":"train",                  #根据这个来决定是训练还是交易
        "random_start":True
    }

    #gamma设置为0.9应该能降低收敛的难度，不用看得那么远
    SAC_PARAMS = {
        "batch_size": 1024*8*4*2,      # 一个批次训练的样本数量
        "buffer_size": 100000,
        "learning_rate": 0.0002,
        "action_noise": "normal",
        "gradient_steps": 500,       # 一共训练多少个批次，一共看了一千万次，平均每个样本看100次
        "train_freq": (5000, "step"),  # 采样多少次训练一次，buff是100000，基本每2次要换全部样本.4个线程，4万次才训练一次
        "learning_starts": 10
    }

    POLICY_KWARGS = dict(net_arch=dict(pi=[256, 128, 128, 64, 64], qf=[256, 128, 128, 64, 64]))

    if platform.system() == 'Windows':
        total_timesteps = 50000  # 总的采样次数,不能太少
        env_kwargs_train = {
            "stock_dim": stock_dimension,
            "hmax": 1000,
            "initial_amount": 1000000,                            #多准备点金钱，让ai能够频繁买卖
            "buy_cost_pct": 6.87e-5,
            "sell_cost_pct": 1.0687e-3,
            "reward_scaling": 1e-1,
            "state_space": state_space,
            "action_space": stock_dimension,
            "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
            "print_verbosity": 100,
            "initial_buy": True,
            "hundred_each_trade": True,
            "out_of_cash_penalty": 0.001,
            "cash_limit": 0.2,
            "model_name":"stock_a",
            "mode":"train",                  #根据这个来决定是训练还是交易
            "random_start":True
        }

        SAC_PARAMS = {
            "batch_size": 1024*8*4*2,                    #一个批次训练的样本数量
            "buffer_size": 100000,                   #每个看1000次，需要1亿次
            "learning_rate": 0.0002,
            "action_noise": "normal",
            "gradient_steps": 500,                  # 一共训练多少个批次
            "train_freq": (5000, "step"),            # 采样多少次训练一次
            "learning_starts": 10
        }

        POLICY_KWARGS = dict(net_arch=dict(pi=[256, 128, 128, 64, 64], qf=[256, 128, 128, 64, 64]))

    print("total_timesteps = {0}".format(total_timesteps))

    e_train_gym = StockTradingEnv(df=train, **env_kwargs_train)

    n_cores = multiprocessing.cpu_count()
    print("core count = {0}".format(n_cores))

    env_train = None

    if platform.system() == 'Windows':
        env_train, _ = e_train_gym.get_multiproc_env(n=n_cores)
    else:
        env_train, _ = e_train_gym.get_multiproc_env(n=n_cores)

    agent = DRLAgent(env=env_train)

    model_sac_before_train = None

    if os.path.exists("moneyMaker_sac.model"):
        model_sac_before_train = SAC.load("moneyMaker_sac.model")
        model_sac_before_train.set_env(env_train)
        model_sac_before_train.load_replay_buffer("moneyMaker_replay_buffer_sac.pkl")
        print("load moneyMaker")
    else:
        model_sac_before_train = agent.get_model("sac", seed=46, model_kwargs=SAC_PARAMS, policy_kwargs=POLICY_KWARGS)
        print("no moneyMaker")

    for i in range(20):
        print("start train")
        model_sac_after_train = agent.train_model(model=model_sac_before_train, tb_log_name='sac',total_timesteps=total_timesteps)
        print("end train")
        model_sac_after_train.save("moneyMaker_sac.model")
        model_sac_after_train.save_replay_buffer("moneyMaker_replay_buffer_sac.pkl")

        env_kwargs_test = {
            "stock_dim": stock_dimension,
            "hmax": 1000,
            "initial_amount": 1000000,
            "buy_cost_pct": 6.87e-5,
            "sell_cost_pct": 1.0687e-3,
            "reward_scaling": 1e-1,
            "state_space": state_space,
            "action_space": stock_dimension,
            "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
            "print_verbosity": 1,
            "initial_buy": False,
            "hundred_each_trade": True,
            "out_of_cash_penalty": 0.001,
            "cash_limit": 0.2,
            "model_name":"stock_a",
            "mode":"test",                  #根据这个来决定是训练还是测试
            "random_start":False
        }

        e_trade_gym = StockTradingEnv(df=trade, **env_kwargs_test)
        print("start test")
        df_account_value, df_actions = DRLAgent.DRL_prediction(model=model_sac_after_train, environment=e_trade_gym)
        print("end test")
        df_actions.to_csv("action.csv", index=False)
        df_account_value.to_csv("account.csv", index=False)
        print("game end")

# 下面代码从网上取的数据有时会异常导致崩溃

'''

# ### Backtest  下面都是画图的操作，pycharm上没效果
plotter = ReturnPlotter(df_account_value, trade, val_start_date, val_stop_date)
plotter.plot_all()  #此接口会从网上取数据，与df_account_value不契合，导致崩溃
plotter.plot()
plotter.plot("000016")
baseline_df = plotter.get_baseline("399300")


daily_return = plotter.get_return(df_account_value)
daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

perf_func = timeseries.perf_stats
perf_stats_all = perf_func(returns=daily_return, factor_returns=daily_return_base, positions=None, transactions=None, turnover_denom="AGB")


print("==============DRL Strategy Stats===========")
with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(returns=daily_return, benchmark_rets=daily_return_base, set_context=False)

print("hahahahahahaha")

'''
