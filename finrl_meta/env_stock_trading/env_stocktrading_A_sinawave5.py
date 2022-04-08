
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
        self.cost_holds = [0]*self.stock_dim                    #个股持仓的总费用，买入卖出会变
        self.cost_friction = 0                                  #摩擦费用
        self.count_0 = 0                                        #为了提高采样的质量，记录无操作的次数

        self.actions_memory=[]
        self.date_memory=[]
        self.asset_memory=[]


    def reset(self):
        if self.mode == 'train':
            lll = len(self.df.date.unique())
            length = int(lll*0.95)
            day_start = random.choice(range(length))
            self.day_start = 0
        else:
            self.day_start = 0

        print("day_start {0}".format(self.day_start))
        self.day = self.day_start

        self.cash = self.initial_amount                         #现金.如果是train，应该根据域范围随机得到

        self.holds = [0]*self.stock_dim                         #持仓
        self.cost_holds = [0]*self.stock_dim
        self.cost_friction = 0                                  #摩擦费用
        self.count_0 = 0

        self.actions_memory=[]
        self.date_memory=[]
        self.asset_memory=[]
        self.cash_memory = []

        self.date_memory.append(self._get_date())
        self.asset_memory.append(self.cash)
        self.cash_memory.append(self.cash)

        #if self.mode == 'train':
            #self._initial_cash_and_buy_()

        state = self._update_state()
        return state

    def _initial_cash_and_buy_(self):
        """Initialize the state, already bought some"""
        data = self.df.loc[self.day, :]

        '''
        cash_max = max(data.cash_max)
        cash_min = min(data.cash_min)

        if cash_max > 10000*10:
            cash_max = 10000*10
        if cash_min < 10000*0.1:
            cash_min = 10000*0.1

        cash_u = random.uniform(cash_min, cash_max)

        self.cash = self.initial_amount/10000 * cash_u
        '''


        prices = data.close.values.tolist()
        avg_price = sum(prices)/len(prices)
        ran = random.random()                 #随机买。因为开始日期是随机的，initial_amount也可以是随机的。需要新加域，表明当前的cash范围,然后在范围内随机一个值
        buy_nums_each_tic = ran*self.cash//(avg_price*len(prices))  # only use half of the initial amount
        buy_nums_each_tic = buy_nums_each_tic//100*100
        cost = sum(prices)*buy_nums_each_tic

        self.cash = self.cash - cost
        self.holds = [buy_nums_each_tic]*self.stock_dim

        '''
        state = [self.initial_amount-cost] + \
                self.data.close.values.tolist() + \
                [buy_nums_each_tic]*self.stock_dim  + \
                sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
        '''

    def step(self, actions):
        #print('step')


        #actions = actions * self.hmax #actions initially is scaled between 0 to 1
        #actions = (actions.astype(int)) #convert into integer because we can't by fraction of shares

        actions_old = None
        if self.mode == 'test':
            actions_old = actions.copy()

        begin_total_asset = self._update_total_assets()

        stocks_can_buy = self._get_can_buy()
        stocks_can_sell = -np.array(self.holds)

        base_ = np.array([-1]*self.stock_dim)

        actions = (actions - base_)/2*(stocks_can_buy - stocks_can_sell)+stocks_can_sell

        argsort_actions = np.argsort(actions)  #索引排序

        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]        #得到卖的索引
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]   #得到买的索引


        reward_sell_all = 0
        for index in sell_index:
            actions[index], rsa = self._sell_stock(index, actions[index])
            actions[index] *= -1
            reward_sell_all += rsa
            #print('rsa {0}'.format(rsa))

        #if reward_sell_all:
            #print('reward_sell_all {0}'.format(reward_sell_all))

        for index in buy_index:
            actions[index] = self._buy_stock(index, actions[index])
        #self.actions_memory.append(actions)                         #此处的action是被处理过的。如果action始终为0也要被惩罚,这属于reword塑形

        self.day += 1

        terminal = self.day >= len(self.df.index.unique())-1

        #if terminal == True:  #统计非0的操作数量
            #count_non0 = np.count_nonzero(self.actions_memory)
            #print('no zero count {0} mode {1}'.format(count_non0, self.mode))


        state = self._update_state()                               #新的一天，close和技术指标都变了

        end_total_asset = self._update_total_assets()

        if self.mode == 'test':
            actions_all = np.hstack((actions, actions_old))
            self.actions_memory.append(actions_all)
            self.date_memory.append(self._get_date())
            self.asset_memory.append(end_total_asset)
            self.cash_memory.append(self.cash)


        #reward = end_total_asset - begin_total_asset

        if terminal == True:            #剩余的，按照最后一天价格全部卖掉
            for index in range(self.stock_dim):
                _, rsa = self._sell_stock(index, self.holds[index])
                reward_sell_all += rsa
            print("sell residual")

        reward = reward_sell_all

        '''
        penalty2 = 0
        if self.cash < end_total_asset*self.cash_limit:        #如果金钱太少，需要进行惩罚，否则在训练的时候因为没钱导致探索空间不够，，训练出来的AI像个傻子，test可以把限制去掉。
            penalty2 = self.initial_amount*self.out_of_cash_penalty
        reward -= penalty2
        '''
        
        '''
        if self.mode == 'train':                       #为了加快采样的有效率。当loss降到一定程度，可以把这块代码注释掉
            count_non0 = np.count_nonzero(actions)
            if count_non0 == 0:
                self.count_0 += 1
                day_pass = self.day - self.day_start
                if self.count_0 > 200:                           #0.99的200次方是0.13，以后把gamma设置小点
                    terminal = True
                    print('terminal by hand')
            else:
                self.count_0 = 0
        '''

        reward = reward * self.reward_scaling

        return state, reward, terminal, {}




    def _sell_stock(self, index, action):
        def _do_sell_normal():
            data = self.df.loc[self.day, :]
            data = data.reset_index(drop=True)
            close = data.close
            price = close[index]
            reward_sell = 0                                      #卖股票赚的钱
            if price > 0:                                        #价格大于0
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.holds[index] > 0:                   #股份大于0

                    cost_avg = self.cost_holds[index]/self.holds[index]            #平均每股成本
                    # Sell only if current asset is > 0
                    sell_num_shares = min(abs(action), self.holds[index])          #不能卖空

                    sell_num_shares = sell_num_shares//100*100                                 #100倍数


                    sell_amount = price * sell_num_shares * (1- self.sell_cost_pct)                #扣除费用，实际获得金额
                    self.cash += sell_amount                                                       #更新金额
                    self.holds[index] -= sell_num_shares                                           #更新股票



                    self.cost_holds[index] = cost_avg * self.holds[index]                          #新的总成本
                    self.cost_friction += price * sell_num_shares * self.sell_cost_pct             #更新交易摩擦费用
                    reward_sell = sell_amount - cost_avg*sell_num_shares                           #获得金额减去成本=赚的
                    #self.trades+=1                                                                #更新交易数量
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares, reward_sell

        sell_num_shares, reward_sell = _do_sell_normal()

        return sell_num_shares, reward_sell

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

                self.cost_holds[index] += buy_amount                                       #买入花费
                self.cost_friction += price * buy_num_shares * self.buy_cost_pct           #更新交易摩擦费用
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

        days_left = self._get_days_left()

        cash = self.cash/self.initial_amount

        holds = np.array(self.holds)/10000

        data0 = self.df.loc[0, :]
        data = self.df.loc[self.day, :]
        close = np.array(data.close)/np.array(data0.close)

        #stock_can_buy = self._get_can_buy()/10000

        cost_holds = np.array(self.cost_holds)/self.initial_amount

        state = np.hstack(
            (
                days_left,
                cash,
                #stock_can_buy,
                cost_holds,
                -holds,
                close
            )
        )
        return state

    def _get_days_left(self):
        day_length = len(self.df.date.unique())
        days_left = day_length - self.day
        return days_left

    def _get_can_buy(self):
        cash_avrage = self.cash/self.stock_dim
        #stock_can_buy = [0]*self.stock_dim
        data = self.df.loc[self.day, :]
        close = np.array(data.close)
        stock_can_buy = cash_avrage/close
        stock_can_buy = stock_can_buy//100*100
        #print('_get_call_buy')
        return stock_can_buy

    def _get_date(self):
        data = self.df.loc[self.day, :]
        date = data.date.unique()[0]
        return date

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        cash_list = self.cash_memory

        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list, 'cash':cash_list})
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