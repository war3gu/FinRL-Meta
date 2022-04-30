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
        #self.cost_friction = 0                                  #摩擦费用

        self.stock_fail_count = [0]*self.stock_dim             #每个回合失败的股票操作数目

        self.stock_succ_count = [0]*self.stock_dim

        self.total_assets = 0

        self.actions_memory=[]
        self.date_memory=[]
        self.asset_memory=[]
        self.reward_memory=[]

        self.ppp = False


    def reset(self):
        if self.mode == 'train':
            lll = len(self.df.date.unique())
            length = int(lll*0.90)
            day_start = random.choice(range(length))
            self.day_start = 4
        else:
            self.day_start = 4

        #print("day_start {0}".format(self.day_start))
        self.day = self.day_start

        self.cash = self.initial_amount                         #现金.如果是train，应该根据域范围随机得到

        self.holds = [0]*self.stock_dim                         #持仓
        self.cost_holds = [0]*self.stock_dim
        #self.cost_friction = 0                                  #摩擦费用
        self.stock_fail_count = [0]*self.stock_dim              #每个回合失败的股票操作数目

        self.stock_succ_count = [0]*self.stock_dim

        self.total_assets = 0

        self.actions_memory=[]
        self.date_memory=[]
        self.asset_memory=[]
        self.cash_memory = []
        self.reward_memory = []

        self.date_memory.append(self._get_date())
        self.asset_memory.append(self.cash)
        self.cash_memory.append(self.cash)

        ran = random.random()
        self.ppp = False
        if ran < 0.05:
            self.ppp = True


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
        #ran = 0.5
        buy_nums_each_tic = ran*self.cash//(avg_price*len(prices))  # only use half of the initial amount
        buy_nums_each_tic = buy_nums_each_tic#//100*100
        cost = sum(prices)*buy_nums_each_tic

        self.cash = self.cash - cost
        self.holds = [buy_nums_each_tic]*self.stock_dim
        #self.cost_holds = prices * buy_nums_each_tic
        self.cost_holds = [price*buy_nums_each_tic for price in prices]

        '''
        state = [self.initial_amount-cost] + \
                self.data.close.values.tolist() + \
                [buy_nums_each_tic]*self.stock_dim  + \
                sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
        '''

    def step(self, actions):
        #print('step')

        #self.cost_friction = 0


        #actions = actions * self.hmax #actions initially is scaled between 0 to 1
        #actions = (actions.astype(int)) #convert into integer because we can't by fraction of shares

        #print('day {0}'.format(self.day))

        #3个action表示最新的持仓比例，不会出现一个action导致另一个无效，也不会一个action本身就无效

        begin_total_asset = self._update_total_assets()

        data = self.df.loc[self.day, :]
        data = data.reset_index(drop=True)
        close = data.close
        avg_price = sum(close)/len(close)

        actions_old = actions.copy()

        assets_ratio = actions[0:2]

        '''
        double = 5
        #cash_ratio = math.exp(double*assets_ratio[2])/(math.exp(double*assets_ratio[0])+math.exp(double*assets_ratio[1])+math.exp(double*assets_ratio[2]))

        ratio_base = 10**assets_ratio[0] - 1 + 10**assets_ratio[1] - 1 + 1

        stock0_ratio = (10**assets_ratio[0] - 1)/ratio_base
        stock1_ratio = (10**assets_ratio[1] - 1)/ratio_base


        #stock0_ratio = assets_ratio[0]
        #stock1_ratio = (1-stock0_ratio)*assets_ratio[1]


        #cash_amount = begin_total_asset*cash_ratio          #用不到
        stock0_amount = begin_total_asset*stock0_ratio/close[0]
        stock1_amount = begin_total_asset*stock1_ratio/close[1]
        '''

        stock0_amount_exchange = self.hmax * assets_ratio[0]
        stock1_amount_exchange = self.hmax * assets_ratio[1]

        #if self.ppp:
            #print('stock_amount_exchange {0}        {1}'.format(stock0_amount_exchange, stock1_amount_exchange))

        #if self.day == 100:
            #print('100')

        reward_sell_all = 0
        reward_buy_all  = 0
        share_sell_all = 0
        share_buy_all = 0

        if stock0_amount_exchange < 0:
            ssa0, rsa0 = self._sell_stock(0, stock0_amount_exchange)
            reward_sell_all += rsa0
            share_sell_all  += ssa0

        if stock1_amount_exchange < 0:
            ssa1, rsa1 = self._sell_stock(1, stock1_amount_exchange)
            reward_sell_all += rsa1
            share_sell_all  += ssa1

        if stock0_amount_exchange >= 0:
            sba0, rba0 = self._buy_stock(0, stock0_amount_exchange)
            reward_buy_all += rba0
            share_buy_all  += sba0

        if stock1_amount_exchange >= 0:
            sba1, rba1 = self._buy_stock(1, stock1_amount_exchange)
            reward_buy_all += rba1
            share_buy_all  += sba1




        '''
        buy0_stock = buy0_money/close[0]
        buy1_stock = buy1_money/close[1]

        sell0_stock = self.holds[0]*sell_ratio[0]
        sell1_stock = self.holds[1]*sell_ratio[1]

        reward_sell_all = 0
        reward_buy_all  = 0

        share_sell_all = 0
        share_buy_all = 0


        
        order = self._get_Order()

        if order == 0:                      #先买后卖
            #print('haha')
            sba0, rba0 = self._buy_stock(0, buy0_stock)
            reward_buy_all += rba0
            share_buy_all += sba0
            sba1, rba1 = self._buy_stock(1, buy1_stock)
            reward_buy_all += rba1
            share_buy_all  += sba1

            ssa0, rsa0 = self._sell_stock(0, sell0_stock)
            reward_sell_all += rsa0
            share_sell_all += ssa0
            ssa1, rsa1 = self._sell_stock(1, sell1_stock)
            reward_sell_all += rsa1
            share_sell_all += ssa1
        else:                               #先卖后买
            #print('xixi')
            ssa0, rsa0 = self._sell_stock(0, sell0_stock)
            reward_sell_all += rsa0
            share_sell_all += ssa0
            ssa1, rsa1 = self._sell_stock(1, sell1_stock)
            reward_sell_all += rsa1
            share_sell_all += ssa1

            sba0, rba0 = self._buy_stock(0, buy0_stock)
            reward_buy_all += rba0
            share_buy_all += sba0
            sba1, rba1 = self._buy_stock(1, buy1_stock)
            reward_buy_all += rba1
            share_buy_all += sba1
        '''


        self.day += 1

        terminal = self.day >= len(self.df.index.unique())-1


        #state = self._update_state()                               #新的一天，close和技术指标都变了

        end_total_asset = self._update_total_assets()

        if self.mode == 'test':
            actions_all = np.hstack((actions, actions_old))
            self.actions_memory.append(actions_all)
            self.date_memory.append(self._get_date())
            self.asset_memory.append(end_total_asset)
            self.cash_memory.append(self.cash)

        reward_end = 0
        if terminal == True:            #剩余的，按照最后一天价格全部卖掉
            for index in range(self.stock_dim):
                _, rsa = self._sell_stock(index, self.holds[index])
                #reward_sell_all += rsa                                #导致异常?
                reward_sell_all += 0
            reward_end = end_total_asset - self.initial_amount         #最后的盈利算入reward，会慢慢传到到前面的状态


        #change = end_total_asset - begin_total_asset


        #reward_buy_all = -reward_buy_all  #买不算摩擦费用，买的摩擦费用算在卖中,鼓励交易.

        #reward_buy_all = -reward_buy_all

        reward_stock_fail = (self.stock_fail_count[0]*close[0] + self.stock_fail_count[1]*close[1])*(-1e-4)  #这个与action密切相关，noise变了，这个也变了

        reward_stock_succ = (self.stock_succ_count[0]*close[0] + self.stock_succ_count[1]*close[1])*(1e-4)

        #操作失误的惩罚带入状态
        '''
        if self.cash >= abs(reward_stock_fail):
            self.cash += reward_stock_fail
        elif self.cost_holds[0] >= abs(reward_stock_fail):
            self.cost_holds[0] -= reward_stock_fail
        elif self.cost_holds[1] >= abs(reward_stock_fail):
            self.cost_holds[1] -= reward_stock_fail
        else:
            print("fuck game over")
        '''


        reward = reward_sell_all + reward_buy_all# + reward_stock_fail + reward_stock_succ  # - self.cost_friction - self.initial_amount*0.001       #需要减去摩擦费用，防止频繁交易

        #if self.ppp:
            #print('stock_fail_count {0}        {1}'.format(self.stock_fail_count[0], self.stock_fail_count[1]))
            #print('stock_succ_count {0}        {1}'.format(self.stock_succ_count[0], self.stock_succ_count[1]))

        #if share_buy_all + share_sell_all <= 100:                        #无操作，cash算利息，防止ai摆烂，因为买的收益为负
        #reward += -self.cash * 1e-4                                  #不用cash试试

        #在股票为空的情况下，无论怎么卖都是一样的结果
        #在cash为空的情况下，无论怎么买都是一样的结果


        self.reward_memory.append(reward)

        earn1 = None
        if terminal == True:
            earn1 = self.cash/self.initial_amount - 1
            #print("sell residual earn1 = {0} \n".format(earn1))

            earn2 = np.sum(self.reward_memory)
            earn2 = earn2/self.initial_amount
            #print("sell residual earn2 = {0} \n".format(earn2))

            #print('mode {0} \n'.format(self.mode))

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

        self.stock_fail_count = [0]*self.stock_dim

        self.stock_succ_count = [0]*self.stock_dim

        #if self.mode == 'train':
            #penalty = 1000000*action_illeagl                   #重重的惩罚
            #cash是不是也要相应的处理
            #if self.cash > penalty:
                #self.cash -= penalty
            #reward -= penalty * self.reward_scaling


        #if reward < 0:
            #reward *= 10


        #if self.cash < end_total_asset*self.cash_limit:
            #reward -= self.initial_amount*0.0001*self.reward_scaling    #无差别减去等于没减
        #elif self.cash > end_total_asset*(1-self.cash_limit):
            #reward -= self.initial_amount*0.0001*self.reward_scaling


        #self.action_space.low = -np.array(self.holds)

        #stocks_can_buy = self._get_can_buy()

        #self.action_space.high = np.array(stocks_can_buy)

        #action_space = self.action_space

#当前第几天告诉callback，callback根据天数，修改sigma

        #if reward < -500:
            #print('fuck stock')

        state = self._update_state()

        return state, reward, terminal, {'earn1':earn1}




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

                    sell_num_shares = sell_num_shares#//100*100                                 #100倍数

                    if abs(action) > self.holds[index]:                                                #不够卖
                        self.stock_fail_count[index] += abs(sell_num_shares - abs(action))        #记录一下操作失败的股票数目

                    self.stock_succ_count[index] += sell_num_shares

                    sell_amount = price * sell_num_shares * (1- self.sell_cost_pct)                #扣除费用，实际获得金额
                    self.cash += sell_amount                                                       #更新金额
                    self.holds[index] -= sell_num_shares                                           #更新股票



                    self.cost_holds[index] = cost_avg * self.holds[index]                          #新的总成本
                    #self.cost_friction += price * sell_num_shares * self.sell_cost_pct             #更新交易摩擦费用
                    #reward_sell = sell_amount - cost_avg*sell_num_shares                           #卖的收益可正可负，考虑了摩擦成本

                    reward_sell = (price - cost_avg)*sell_num_shares
                else:
                    self.stock_fail_count[index] += abs(action)                    #小心此处
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
            reward_buy = 0
            if price > 0:                                                                 #股票价格大于0
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.cash // price                                    #所有钱能买的数量

                avg_price = price

                if self.holds[index]:
                    avg_price = self.cost_holds[index]/self.holds[index]

                # update balance
                buy_num_shares = min(available_amount, action)                               #实际能买的数量
                buy_num_shares = buy_num_shares#//100*100

                if abs(action) > buy_num_shares:                                                     #钱不够买
                    self.stock_fail_count[index] += abs(buy_num_shares - abs(action))           #记录一下操作失败的股票数目

                self.stock_succ_count[index] += buy_num_shares

                buy_amount = price * buy_num_shares * (1 + self.buy_cost_pct)              #实际花费的金额
                self.cash -= buy_amount                                                    #更新金额

                self.holds[index] += buy_num_shares                                        #更新股票数量

                self.cost_holds[index] += buy_amount                                       #买入花费
                #self.cost_friction += price * buy_num_shares * self.buy_cost_pct           #更新交易摩擦费用
                #reward_buy = -price * buy_num_shares * self.buy_cost_pct                   #买的收益为负数,包含了摩擦费用

                reward_buy = -(price - avg_price)*buy_num_shares
            else:
                buy_num_shares = 0

            return buy_num_shares, reward_buy

        buy_num_shares, reward_buy = _do_buy()

        return buy_num_shares, reward_buy

    def _get_Order(self):
        total_assets = self._update_total_assets()
        if self.cash > total_assets*0.5:
            return 0
        else:
            return 1

    def _update_total_assets(self):
        data = self.df.loc[self.day, :]
        close = data.close
        total_assets = self.cash + sum(np.array(close)*np.array(self.holds))
        #print('_update_total_assets')
        self.total_assets = total_assets/self.initial_amount
        return total_assets

    def _update_state(self):

        #days_left = self._get_days_left()

        cash = self.cash/self.initial_amount

        holds = np.array(self.holds)/10000

        data0 = self.df.loc[0, :]
        data = self.df.loc[self.day-4:self.day, :]
        #m1 = data.close.unique()
        #m2 = data0.close.unique()
        close = np.array(data.close)/np.array(data0.close).sum()

        #stock_can_buy = self._get_can_buy()/10000

        cost_holds = np.array(self.cost_holds)/self.initial_amount

        #stock_fail_count = np.array(self.stock_fail_count)/1000

        #order = self._get_Order()

        state = np.hstack(
            (
                #days_left,
                cash,
                #stock_can_buy,
                cost_holds,
                -holds,
                close,
                #order
                #stock_fail_count,
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
        stock_can_buy = stock_can_buy#//100*100
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