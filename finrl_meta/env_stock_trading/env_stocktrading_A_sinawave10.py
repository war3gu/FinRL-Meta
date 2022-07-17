import math

import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import Logger as log

import random

from copy import deepcopy

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 paths,
                 #stock_dim,
                 hmax,
                 initial_amount,
                 buy_cost_pct,
                 sell_cost_pct,
                 reward_scaling,
                 state_space,
                 action_space,
                 mode='',
                 out_of_cash_penalty=0.01,
                 cash_limit=0.1):
        self.paths = paths                                      #所有的训练路径
        self.path_index = -1                                    #路径索引
        self.df = None                                          #数据
        #self.stock_dim = stock_dim                              #股票数量
        self.hmax = hmax                                        #每日最大交易数量
        self.initial_amount = initial_amount                    #启动资金
        self.buy_cost_pct = buy_cost_pct                        #买摩擦费用
        self.sell_cost_pct = sell_cost_pct                      #卖摩擦费用
        self.reward_scaling = reward_scaling                    #奖励放大倍数
        self.state_space = state_space                          #状态维度
        self.action_space = action_space                        #操作维度
        self.mode=mode                                          #模式 'test'  'train'
        self.out_of_cash_penalty = out_of_cash_penalty          #资金太少的惩罚
        self.cash_limit = cash_limit                            #资金极限占比

###################################################################################
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space,))
####################################################################################

    def get_path_length(self):
        return len(self.paths)

    def increase_Path_Index(self):
        self.path_index += 1                                    #当前训练的是哪个episode
        self.path_index  = self.path_index % len(self.paths)


    def reset(self):
        self.increase_Path_Index()

        self.df = self.paths[self.path_index]
        self.day_start = 0
        self.day = self.day_start
        self.cash = self.initial_amount                         #现金.如果是train，应该根据域范围随机得到
        self.holds = 0                                          #持仓
        #self.cost_friction = 0                                 #交易总的摩擦费用
        #self.total_assets = 0


        if self.mode == "test":           #test结束，架构会自动执行reset，导致交易信息丢失。（此处让reset基本什么都不做，解决问题）
            state = self._update_state()
        else:
            state = self._update_state()
        return state

    def reset_memory(self):
        self.actions_memory=[]
        self.date_memory=[]
        self.asset_memory=[]
        self.cash_memory = []
        self.holds_memory = []

        #self.reward_memory = []

        self.date_memory.append(self._get_date())
        self.asset_memory.append(self.cash)
        self.cash_memory.append(self.cash)
        self.holds_memory.append(self.holds)

    '''
    def _initial_cash_and_buy_(self):                           #可以买空卖空，不需要初始化买入股票了
        """Initialize the state, already bought some"""
        data = self.df.loc[self.day, :]

        prices = data.close.values.tolist()
        avg_price = sum(prices)/len(prices)
        ran = random.random()
        buy_nums_each_tic = ran*self.cash//(avg_price*len(prices))  # only use half of the initial amount
        buy_nums_each_tic = buy_nums_each_tic#//100*100
        cost = sum(prices)*buy_nums_each_tic

        self.cash = self.cash - cost
        self.holds = [buy_nums_each_tic]*self.stock_dim
    '''


    def step(self, actions):
        #print('step')

        begin_total_asset = self._update_total_assets()

        #data = self.df.loc[self.day, :]
        #data = data.reset_index(drop=True)

        actions_old = actions.copy()

        assets_ratio = actions[0:2]

        stock0_amount_exchange = self.hmax * assets_ratio[0]


        if stock0_amount_exchange < 0:
            ssa0 = self._sell_stock(0, stock0_amount_exchange)

        if stock0_amount_exchange >= 0:
            sba0 = self._buy_stock(0, stock0_amount_exchange)


        self.day += 1

        terminal = self.day >= len(self.df.index.unique())-1

        state = self._update_state()                               #新的一天，close和技术指标都变了

        end_total_asset = self._update_total_assets()

        if self.mode == 'test':
            #actions_all = np.hstack((actions, actions_old))
            actions_all = np.hstack((actions))
            self.actions_memory.append(actions_all)
            self.date_memory.append(self._get_date())
            self.asset_memory.append(end_total_asset)
            self.cash_memory.append(self.cash)
            self.holds_memory.append(self.holds)



        reward = end_total_asset - begin_total_asset


        #在股票为空的情况下，无论怎么卖都是一样的结果
        #在cash为空的情况下，无论怎么买都是一样的结果
        #TD3的探索有问题


        #self.reward_memory.append(reward)

        earn1 = None
        if terminal == True:
            earn1 = end_total_asset - self.initial_amount
            earn1 *= self.reward_scaling
            print('earn1 = {0}'.format(earn1))
            #self.increase_Path_Index()

        reward = reward * self.reward_scaling

        return state, reward, terminal, {'earn1':earn1}




    def _sell_stock(self, index, action):
        def _do_sell_normal():
            data = self.df.loc[self.day, :]
            #data = data.reset_index(drop=True)
            close = data.close
            price = close
            sell_num_shares = 0
            if price > 0:                                                                      #价格大于0
                sell_num_shares    = abs(action)                                               #可以卖空，正数
                sell_amount        = price * sell_num_shares * (1- self.sell_cost_pct)         #扣除费用，实际获得金额
                self.cash         += sell_amount                                               #更新金额
                self.holds        -= sell_num_shares                                           #更新股票数量，可能为负数

                #self.cost_friction += price * sell_num_shares * self.sell_cost_pct            #更新交易摩擦费用

            return sell_num_shares

        sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):

        def _do_buy():
            data = self.df.loc[self.day, :]
            #data = data.reset_index(drop=True)
            close = data.close
            price = close

            if price > 0:                                                                      #股票价格大于0
                buy_num_shares     = action                                                    #可以买空，正数
                buy_amount         = price * buy_num_shares * (1 + self.buy_cost_pct)          #实际花费的金额
                self.cash         -= buy_amount                                                #更新金额，可能为负数
                self.holds        += buy_num_shares                                            #更新股票数量

                #self.cost_friction += price * buy_num_shares * self.buy_cost_pct              #更新交易摩擦费用

            return buy_num_shares

        buy_num_shares = _do_buy()

        return buy_num_shares

    def _get_Order(self):
        total_assets = self._update_total_assets()
        if self.cash > total_assets*0.5:
            return 0
        else:
            return 1

    def _update_total_assets(self):
        data = self.df.loc[self.day, :]
        close = data.close
        total_assets = self.cash + close*self.holds
        #print('_update_total_assets')
        #self.total_assets = total_assets
        return total_assets

    def _update_state(self):
        cash = self.cash/10000

        holds = np.array(self.holds)/1000

        data0 = self.df.loc[0, :]
        #data = self.df.loc[self.day-self.day_start:self.day, :]
        data = self.df.loc[self.day, :]
        #close = np.array(data.close)/np.array(data0.close).sum()

        close = data.close


        state = np.hstack(
            (
                #self.path_index,
                #cash,
                holds,
                close,
                self.day
            )
        )
        return state

    def _get_days_left(self):
        day_length = len(self.df.date.unique())
        days_left = day_length - self.day
        return days_left

    '''
    def _get_can_buy(self):
        cash_avrage = self.cash/self.stock_dim
        #stock_can_buy = [0]*self.stock_dim
        data = self.df.loc[self.day, :]
        close = np.array(data.close)
        stock_can_buy = cash_avrage/close
        stock_can_buy = stock_can_buy#//100*100
        #print('_get_call_buy')
        return stock_can_buy
    '''

    def _get_date(self):
        data = self.df.loc[self.day, :]
        #date = data.date.unique()[0]
        date = data.date
        return date

    def get_step_length(self):
        lll = len(self.df.index.unique())
        return lll - self.day_start

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        cash_list = self.cash_memory
        holds_list = self.holds_memory

        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list, 'cash':cash_list, 'holds':holds_list})
        return df_account_value

    def save_action_memory(self):
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        #df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        #obs = e.reset()
        return e, None

    def get_multiproc_env(self, n=10):
        def get_self():
            return deepcopy(self)
        e = SubprocVecEnv([get_self for _ in range(n)], start_method=None)  #Only ‘forkserver’ and ‘spawn’ start methods are thread-safe
        #obs = e.reset()
        return e, None