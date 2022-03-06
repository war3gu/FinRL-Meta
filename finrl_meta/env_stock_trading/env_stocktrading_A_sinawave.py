
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
                 df,
                 stock_dim,
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

        self.df = df                                            #数据
        self.stock_dim = stock_dim                              #股票数量
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

        self.day_start = 0                                      #开始日期
        self.day = self.day_start                               #当前日期
        self.cash = self.initial_amount                         #现金
        self.holds = [0]*self.stock_dim                         #持仓
        self.cost = 0

        self.actions_memory=[]
        self.date_memory=[]
        self.asset_memory=[]


    def reset(self):
        if self.mode == 'train':
            lll = len(self.df.date.unique())
            length = int(lll*0.01)
            day_start = random.choice(range(length))
            self.day_start = day_start
        else:
            self.day_start = 0
        self.day = self.day_start
        self.cash = self.initial_amount                         #现金
        self.holds = [0]*self.stock_dim                         #持仓
        self.cost = 0

        self.actions_memory=[]
        self.date_memory=[]
        self.asset_memory=[]

        self.date_memory.append(self._get_date())
        self.asset_memory.append(self.cash)

        state = self._update_state()
        return state

    def step(self, actions):
        #print('step')


        actions = actions * self.hmax #actions initially is scaled between 0 to 1
        actions = (actions.astype(int)) #convert into integer because we can't by fraction of shares

        begin_total_asset = self._update_total_assets()

        argsort_actions = np.argsort(actions)

        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]



        for index in sell_index:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)

        for index in buy_index:
            actions[index] = self._buy_stock(index, actions[index])
        #self.actions_memory.append(actions)                         #此处的action是被处理过的。如果action始终为0也要被惩罚,这属于reword塑形

        self.day += 1

        terminal = self.day >= len(self.df.index.unique())-1

        #if terminal == True:
            #print('hahahaha')


        state = self._update_state()                               #新的一天，close和技术指标都变了

        end_total_asset = self._update_total_assets()

        self.actions_memory.append(actions)
        self.date_memory.append(self._get_date())
        self.asset_memory.append(end_total_asset)

        reward = end_total_asset - begin_total_asset                #总资产差就是reward

        '''
        penalty2 = 0
        if self.cash < end_total_asset*self.cash_limit:        #如果金钱太少，需要进行惩罚，否则在训练的时候因为没钱导致探索空间不够，，训练出来的AI像个傻子，test可以把限制去掉。
            penalty2 = self.initial_amount*self.out_of_cash_penalty
        reward -= penalty2
        '''

        reward = reward * self.reward_scaling

        return state, reward, terminal, {}




    def _sell_stock(self, index, action):
        def _do_sell_normal():
            data = self.df.loc[self.day, :]
            data = data.reset_index(drop=True)
            close = data.close
            price = close[index]
            if price > 0:                                        #价格大于0
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.holds[index] > 0:                   #股份大于0
                    # Sell only if current asset is > 0
                    sell_num_shares = min(abs(action), self.holds[index])          #不能卖空

                    sell_num_shares = sell_num_shares//100*100                                 #100倍数


                    sell_amount = price * sell_num_shares * (1- self.sell_cost_pct)                #扣除费用，实际获得金额
                    self.cash += sell_amount                                                       #更新金额
                    self.holds[index] -= sell_num_shares                                           #更新股票
                    self.cost += price * sell_num_shares * self.sell_cost_pct                      #更新交易摩擦费用
                    #self.trades+=1                                                                #更新交易数量
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):

        def _do_buy():
            data = self.df.loc[self.day, :]
            data = data.reset_index(drop=True)
            close = data.close
            price = close[index]
            if price > 0:                                                                 #股票价格大于0
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.cash // price                                    #所有钱能买的数量

                # update balance
                buy_num_shares = min(available_amount, action)                               #实际能买的数量
                buy_num_shares = buy_num_shares//100*100

                buy_amount = price * buy_num_shares * (1 + self.buy_cost_pct)              #实际花费的金额
                self.cash -= buy_amount                                                    #更新金额

                self.holds[index] += buy_num_shares                                        #更新股票数量

                self.cost += price * buy_num_shares * self.buy_cost_pct                    #更新交易摩擦费用
                #self.trades+=1                                                            #更新交易数量
            else:
                buy_num_shares = 0

            return buy_num_shares

        buy_num_shares = _do_buy()

        return buy_num_shares


    def _update_total_assets(self):
        data = self.df.loc[self.day, :]
        close = data.close
        total_assets = self.cash + sum(np.array(close)*np.array(self.holds))
        #print('_update_total_assets')
        return total_assets

    def _update_state(self):
        data = self.df.loc[self.day, :]
        close = data.close
        state = np.hstack(
            (
                self.cash,
                self.holds,
                close
            )
        )
        return state

    def _get_date(self):
        data = self.df.loc[self.day, :]
        date = data.date.unique()[0]
        return date

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory

        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
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
        obs = e.reset()
        return e, obs

    def get_multiproc_env(self, n=10):
        def get_self():
            return deepcopy(self)
        e = SubprocVecEnv([get_self for _ in range(n)], start_method=None)  #Only ‘forkserver’ and ‘spawn’ start methods are thread-safe
        obs = e.reset()
        return e, obs