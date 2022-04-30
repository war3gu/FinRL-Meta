
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
from finrl_meta.env_stock_trading.env_stocktrading_A_sinawave8 import StockTradingEnv
from drl_agents.stablebaselines3_models import DRLAgent

pd.options.display.max_columns = None

import pyfolio
from pyfolio import timeseries

import multiprocessing
import platform

from stable_baselines3 import TD3
from stable_baselines3 import SAC

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TKAgg')
import numpy as np
import torch
from torchsummary import summary

print("ALL Modules have been imported!")

# ### Create folders

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print('Using GPU:', torch.cuda.get_device_name('cuda'))

def convert(action):
    if action[0] == 0 and action[1] == 0:
        return ''
    else:
        str = '{0},{1}'.format(action[0], action[1])
        return str

def draw_results(data, actions, account_value):
    newData = data.copy()
    newData = newData.reset_index(drop=True)
    stock1 = newData[(newData['tic'] == 'sinawave_noise1')]
    stock1 = stock1.reset_index(drop=True)
    stock2 = newData[(newData['tic'] == 'sinawave_noise2')]
    stock2 = stock2.reset_index(drop=True)

    length = len(newData.date.unique())
    x = np.arange(0, length)
    plt.xlabel('index')
    plt.ylabel('close')
    plt.grid()
    plt.plot(x, stock1['close'], 'k--')
    plt.plot(x, stock2['close'], 'r-.')

    actions = actions.append([{0:0,1:0}])
    actions = actions.append([{0:0,1:0}])
    newAction = pd.DataFrame(actions.apply(lambda x : convert(x), axis=1))

    y = stock2['close']

    z = newAction.iloc[:, 0]


    for a,b,c in zip(x,y,z):
        if c != '':
            plt.text(a, b, c, fontsize=7)


    #plt.text(100, 100, 'hahahahahaha')

    plt.legend(['sinawave_noise1', 'sinawave_noise2'], loc=1)
    plt.show()

    print('draw_results')
    '''
x = np.arange(0, last, 1/fs)
plt.xlabel('t/s')
plt.ylabel('y')
plt.grid()
plt.plot(x, hz_50, 'k')
plt.plot(x, hz_50_30, 'r-.')
plt.plot(x, hz_50_60, 'g--')
plt.plot(x, hz_50_90, 'b-.')
plt.plot(x, add, 'k')
plt.legend(['phase 0', 'phase 30', 'phase 60', 'phase 90', 'add'], loc=1)
plt.show()
'''



def expandTrain(train):
    print('expandTrain')
    train_expand = None
    train.insert(train.shape[1], 'cash_max', 10000)
    train.insert(train.shape[1], 'cash_min', 10000)
    tic_all = train.tic.unique()

    for tic in tic_all:
        lines = train.loc[train['tic'] == tic]
        line_last = None
        list_all = []
        for index,line in lines.iterrows():
            if line_last is None:
                line['cash_max'] = 10000
                line['cash_min'] = 10000
            else:
                line['cash_max'] = line_last['cash_max']
                line['cash_min'] = line_last['cash_min']
                if line['close'] > line_last['close']:
                    line['cash_max'] *= line['close']/line_last['close']
                    #print('bigger')
                else:
                    line['cash_min'] *= line['close']/line_last['close']
                    #print('smaller')
            list_all.append(line)
            line_last = line
        df_all = pd.DataFrame(list_all)
        if train_expand is None:
            train_expand = df_all
        else:
            train_expand = train_expand.append(df_all)

    #train_expand = pd.DataFrame(list_all)
    train_expand = train_expand.sort_values(['date', "tic"], ignore_index=False)
    print('expandTrain_end')
    return train_expand



if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    token = '27080ec403c0218f96f388bca1b1d85329d563c91a43672239619ef5'
    ts_processor = TushareProProcessor("tusharepro", token=token)

    sina1 = pd.read_csv('datasets/polynomial_noise1.csv')
    sina2 = pd.read_csv('datasets/polynomial_noise2.csv')
    sina = sina1.append(sina2)
    sina = sina.sort_values(['date', "tic"], ignore_index=True)

    train = ts_processor.data_split(sina, '2000-01-01', '2000-05-30')        #短一些，方便训练
    trade = ts_processor.data_split(sina, '2000-01-01', '2000-05-20')
    #trade = ts_processor.data_split(sina, '2000-05-30', '2000-07-18')

    #train = expandTrain(train)

    #draw_results(trade, None, None)

    stock_dimension = len(train.tic.unique())
    state_space = 1 + stock_dimension*6 + stock_dimension   #剩余天数， 现金,持仓*2，股价
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    total_timesteps = 250000  # 总的采样次数,不能太少。一局1000天，相当于玩了1000局，有点少
    #total_timesteps = 2000

    env_kwargs_train = {
        "stock_dim": stock_dimension,
        "hmax": 5000,
        "initial_amount": 100000,                            #多准备点金钱，让ai能够频繁买卖.训练过程中可以慢慢降低这个值
        "buy_cost_pct": 6.87e-5,
        "sell_cost_pct": 1.0687e-3,
        "reward_scaling": 1e-2,
        "state_space": state_space,
        "action_space": 2,                                   #7是3个action
        "out_of_cash_penalty": 0.001,
        "cash_limit": 0.1,
        "mode":"train",                                      #根据这个来决定是训练还是交易
    }

    DDPG_PARAMS = {
        "batch_size": 128*2,                 #一个批次训练的样本数量
        "buffer_size": 300000,                    #每个看1000次，需要1亿次
        "learning_rate": 0.00075,
        "gamma": 0.99,
        "tau": 0.005,                          #0.005
        "target_policy_noise": 0.01,
        "action_noise": "ornstein_uhlenbeck_super",
        "gradient_steps": 100,                     # 一共训练多少个批次,1 - beta1 ** step
        "policy_delay": 2,                        # critic训练多少次才训练actor一次
        "train_freq": (1000, "step"),             # 采样多少次训练一次
        "learning_starts": 10000                  #这个一定要很大，因为AI的初始化输出大多是1，-1
    }

    actor_ratio = 8
    critic_ratio = 8

    POLICY_KWARGS = dict(net_arch=dict(pi=[128*actor_ratio, 512*actor_ratio, 128*actor_ratio], qf=[128*actor_ratio,  512*actor_ratio, 128*actor_ratio]),
                     optimizer_kwargs=dict(weight_decay=0, amsgrad=False, betas=[0.95, 0.99]))

    #POLICY_KWARGS = dict(net_arch=dict(pi=[128*actor_ratio, 512*actor_ratio, 512*actor_ratio, 512*actor_ratio, 128*actor_ratio], qf=[128*critic_ratio, 512*critic_ratio, 512*critic_ratio, 512*critic_ratio, 128*critic_ratio]),
                         #optimizer_kwargs=dict(weight_decay=0, amsgrad=False, betas=[0.95, 0.99]))

    print("total_timesteps = {0}".format(total_timesteps))

    e_train_gym = StockTradingEnv(df=train, **env_kwargs_train)

    n_cores = multiprocessing.cpu_count()
    n_cores = 1
    print("core count = {0}".format(n_cores))

    env_train, _ = e_train_gym.get_multiproc_env(n=n_cores)

    #env_train, _ = e_train_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    model_ddpg_before_train = None

    if os.path.exists("moneyMaker_sina.model"):
        model_ddpg_before_train = TD3.load("moneyMaker_sina.model", custom_objects={'learning_rate': 0.00075, "gamma": 0.99, "batch_size": 128*8, "train_freq": (500, "step"), "gradient_steps": 100}) #必须在此处修改lr
        model_ddpg_before_train.set_env(env_train)

        #dict = model_ddpg_before_train.get_parameters()

        #dict['actor.optimizer']['param_groups'][0]['lr'] = 0.0001           #loss无法下降，修改一下lr试试
        #dict['critic.optimizer']['param_groups'][0]['lr'] = 0.0001

        #model_ddpg_before_train.set_parameters(dict)

        model_ddpg_before_train.load_replay_buffer("moneyMaker_replay_buffer_sina.pkl")
        print("load moneyMaker")
    else:
        model_ddpg_before_train = agent.get_model("td3", seed=46, model_kwargs=DDPG_PARAMS, policy_kwargs=POLICY_KWARGS)
        print("no moneyMaker")

        summary(model_ddpg_before_train.actor, input_size=(1, 1, state_space), batch_size=-1)

    for i in range(20000):
        print("start train")
        model_ddpg_after_train = agent.train_model(model=model_ddpg_before_train, tb_log_name='td3',total_timesteps=total_timesteps)
        print("end train")
        model_ddpg_after_train.save("moneyMaker_sina.model")
        model_ddpg_after_train.save_replay_buffer("moneyMaker_replay_buffer_sina.pkl")

        env_kwargs_test = {
            "stock_dim": stock_dimension,
            "hmax": 1000,
            "initial_amount": 100000,
            "buy_cost_pct": 6.87e-5,
            "sell_cost_pct": 1.0687e-3,
            "reward_scaling": 1e-2,
            "state_space": state_space,
            "action_space": stock_dimension,
            "out_of_cash_penalty": 0.001,
            "cash_limit": 0.2,
            "mode":"test",
        }
        e_trade_gym = StockTradingEnv(df=trade, **env_kwargs_test)
        print("start test")
        df_account_value, df_actions = DRLAgent.DRL_prediction(model=model_ddpg_after_train, environment=e_trade_gym)
        print("end test")
        df_actions.to_csv("action.csv", index=False)
        df_account_value.to_csv("account.csv", index=False)

        #把df_actions显示在图形上，与股价一起
        # #draw_results(trade, df_actions, df_account_value)

    print('sina end')

