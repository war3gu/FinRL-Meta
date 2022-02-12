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
                tech_indicator_list,                 
                turbulence_threshold=None,           
                make_plots = False,                  
                print_verbosity = 2 ,               
                day = 0,                           
                initial=True,                    
                previous_state=[],               
                model_name = '',                  
                mode='',                          
                iteration='',                      
                initial_buy=False,                   # Use half of initial amount to buy
                hundred_each_trade=True,
                out_of_cash_penalty=0.01,
                cash_limit=0.1,
                random_start=True):                  # The number of shares per lot must be an integer multiple of 100

        self.day = day                               
        self.df = df                                 
        self.stock_dim = stock_dim                   
        self.hmax = hmax                            
        self.initial_amount = initial_amount         
        self.buy_cost_pct = buy_cost_pct             
        self.sell_cost_pct = sell_cost_pct          
        self.reward_scaling = reward_scaling         
        self.state_space = state_space              
        self.action_space = action_space           
        self.tech_indicator_list = tech_indicator_list                                                   
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.action_space,))                 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space,))     
        self.data = self.df.loc[self.day,:]        
        self.terminal = False                      
        self.make_plots = make_plots                
        self.print_verbosity = print_verbosity    
        self.turbulence_threshold = turbulence_threshold  
        self.initial = initial                      
        self.previous_state = previous_state      
        self.model_name=model_name               
        self.mode=mode                         
        self.iteration=iteration                   
        # initalize state
        self.initial_buy = initial_buy
        self.hundred_each_trade = hundred_each_trade
        self.out_of_cash_penalty = out_of_cash_penalty
        self.cash_limit = cash_limit
        self.random_start = random_start

        self.state = self._initiate_state()
        
        # initialize reward
        self.reward = 0                              
        self.turbulence = 0                         
        self.cost = 0                           
        self.trades = 0                           
        self.episode = 0                                                 
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]  #总资产
        self.cash_memory  = [self.initial_amount]  #现金
        self.rewards_memory = []                 
        self.actions_memory=[]                       
        self.date_memory=[self._get_date()]
        self.sell_fail_count = 0                   #卖失败次数，需要惩罚
        self.buy_fail_count  = 0                   #买失败次数，需要惩罚
        self._seed()                               
        
        


    def _sell_stock(self, index, action):
        def _do_sell_normal():                    
            if self.state[index+1]>0:                                        #价格大于0
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index+self.stock_dim+1] > 0:                   #股份大于0
                    # Sell only if current asset is > 0
                    sell_num_shares = min(abs(action),self.state[index+self.stock_dim+1])          #不能卖空
                    if self.hundred_each_trade:
                        sell_num_shares = sell_num_shares//100*100                                 #100倍数

                    if sell_num_shares == 0:                                                       #卖0股，要惩罚。防止长时间无操作
                        self.sell_fail_count += 1

                    sell_amount = self.state[index+1] * sell_num_shares * (1- self.sell_cost_pct)  #扣除费用，实际获得金额
                    self.state[0] += sell_amount                                                   #更新金额
                    self.state[index+self.stock_dim+1] -= sell_num_shares                          #更新股票
                    self.cost +=self.state[index+1] * sell_num_shares * self.sell_cost_pct         #更新交易摩擦费用
                    self.trades+=1                                                                 #更新交易数量
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares
            
        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence>=self.turbulence_threshold:
                if self.state[index+1]>0: 
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions 
                    if self.state[index+self.stock_dim+1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index+self.stock_dim+1]
                        sell_amount = self.state[index+1]*sell_num_shares* (1- self.sell_cost_pct)

                        self.state[0] += sell_amount

                        self.state[index+self.stock_dim+1] =0
                        self.cost += self.state[index+1]*self.state[index+self.stock_dim+1]* \
                                    self.sell_cost_pct
                        self.trades+=1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares
   
    def _buy_stock(self, index, action):

        def _do_buy():
            if self.state[index+1]>0:                                                        #股票价格大于0
                # Buy only if the price is > 0 (no missing data in this particular date)       
                available_amount = self.state[0] // self.state[index+1]                      #所有钱能买的数量
                
                # update balance
                buy_num_shares = min(available_amount, action)                               #实际能买的数量
                if self.hundred_each_trade:
                    buy_num_shares = buy_num_shares//100*100

                if buy_num_shares == 0:                                                      #买0股，要惩罚。防止长时间无操作
                    self.buy_fail_count += 1

                buy_amount = self.state[index+1] * buy_num_shares * (1+ self.buy_cost_pct)   #实际花费的金额
                
             
                self.state[0] -= buy_amount                                                  #更新金额

                self.state[index+self.stock_dim+1] += buy_num_shares                         #更新股票数量
                
                self.cost+=self.state[index+1] * buy_num_shares * self.buy_cost_pct          #更新交易摩擦费用
                self.trades+=1                                                               #更新交易数量
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence< self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory,'r')
        plt.savefig('results/account_value_trade_{}.png'.format(self.episode))
        plt.close()

    def step(self, actions):
        self.sell_fail_count = 0
        self.buy_fail_count  = 0
        self.terminal = self.day >= len(self.df.index.unique())-1
        if self.terminal:                              
            print(f"Episode end successful: {self.episode}")
            if self.make_plots:
                self._make_plot()            
            

            end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))   
            df_total_value = pd.DataFrame(self.asset_memory)
            df_cash        = pd.DataFrame(self.cash_memory)
            tot_reward = self.state[0]+sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))- self.initial_amount 

            df_total_value.columns = ['account_value']
            df_total_value['date'] = self.date_memory   
            df_total_value['daily_return']=df_total_value['account_value'].pct_change(1)

            df_cash.columns = ['cash_value']
            df_cash['date'] = self.date_memory

            if df_total_value['daily_return'].std() !=0:
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                      df_total_value['daily_return'].std()
            df_rewards = pd.DataFrame(self.rewards_memory)  
            df_rewards.columns = ['account_rewards']       
            df_rewards['date'] = self.date_memory[:-1]     
            if self.episode % self.print_verbosity == 0:   
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value['daily_return'].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name!='') and (self.mode!=''):  
                df_actions = self.save_action_memory()
                df_actions.to_csv('results/actions_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration))
                df_total_value.to_csv('results/account_value_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration),index=False)
                df_cash.to_csv('results/cash_value_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration),index=False)
                df_rewards.to_csv('results/account_rewards_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration),index=False)
                plt.plot(self.asset_memory,'r')
                plt.savefig('results/account_value_{}_{}_{}.png'.format(self.mode,self.model_name, self.iteration),index=False)
                plt.close()

            # Add outputs to logger interface
            #logger.record(key="environment/portfolio_value", value=end_total_asset)
            #logger.record(key="environment/total_reward", value=tot_reward)
            #logger.record(key="environment/total_reward_pct", value=(tot_reward / (end_total_asset - tot_reward)) * 100)
            #logger.record(key="environment/total_cost", value=self.cost)
            #logger.record(key="environment/total_trades", value=self.trades)

            return self.state, self.reward, self.terminal, {}

        else:

            actions = actions * self.hmax #actions initially is scaled between 0 to 1
            actions = (actions.astype(int)) #convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence>=self.turbulence_threshold:
                    actions=np.array([-self.hmax]*self.stock_dim)
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])) 
            
            begin_close=self.state[1:(self.stock_dim+1)]
            begin_stock=self.state[(self.stock_dim+1):(self.stock_dim*2+1)]        #持有的股票数量
            
            argsort_actions = np.argsort(actions)  
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]         
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]    
            


            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)

            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])
            self.actions_memory.append(actions)          #此处的action是被处理过的。如果action始终为0也要被惩罚,这属于reword塑形

            self.day += 1                                             
            self.data = self.df.loc[self.day,:]                       
            if self.turbulence_threshold is not None:                   
                self.turbulence = self.data['turbulence'].values[0]   
            self.state =  self._update_state()                               #新的一天，close和技术指标都变了

            i_list=[]
            for i in range(self.stock_dim):
                if(begin_stock[i]-self.state[self.stock_dim+1+i]==0):        #某只股票数量没有变化
                    i_list.append(i)               
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])) 
            self.asset_memory.append(end_total_asset)
            self.cash_memory.append(self.state[0])
            self.date_memory.append(self._get_date())                  
            self.reward = end_total_asset - begin_total_asset                #总资产差就是reward


            penalty1 = 0
            penalty2 = 0
            penalty3 = 0
            penalty4 = 0


            for i in i_list:                                                       #无操作需要惩罚
                penalty1 += self.state[i+1]*self.state[self.stock_dim+1+i]*0.001    #此处有个bug，一直没有股票，此处的惩罚是0

            if self.state[0] < end_total_asset*self.cash_limit:        #如果金钱太少，需要进行惩罚，否则在训练的时候因为没钱导致探索空间不够，，训练出来的AI像个傻子，test可以把限制去掉。
                penalty2 = self.initial_amount*self.out_of_cash_penalty*0.1

            penalty3 = end_total_asset*0.00011                         #每天金钱要固定减去一定金额，当做基准利息损失,否则ai会学会什么都不做，0.00011是每天利率

            fail_count = self.sell_fail_count                          #self.buy_fail_count暂时不用,只要金钱留足够就能买
            if fail_count > 0:                                         #15只股票里有10只无操作,惩罚
                penalty4 = end_total_asset*0.0001*fail_count             #0.0001是随便给的，需要调整


            #self.reward = self.reward - penalty2
            #self.reward = -fail_count*1000

            #if self.day % 100 == 0:
                #print("episode = {0} day = {1} fail_count = {2} ".format(self.episode, self.day, fail_count))
                #print("reward penalty1234 = {0} {1} {2} {3} {4}".format(self.reward, penalty1, penalty2, penalty3, penalty4))



            self.rewards_memory.append(self.reward)
            self.reward = self.reward*self.reward_scaling

        if self.mode == "train" and self.state[0] < self.initial_amount*self.out_of_cash_penalty:  #直接结束,这应该是训练的时候可以结束，test的时候不可以结束
            print("episode {0} day {1} day last {2} out of cash".format(self.episode, self.day, self.day-self.day_start))
            return self.state, -end_total_asset*self.cash_limit, True, {}

        return self.state, self.reward, self.terminal, {}

    def reset(self):

        if self.random_start:
            lll = len(self.df.date.unique())
            length = int(lll*0.1)
            day_start = random.choice(range(length))
            self.day_start = day_start
            print("day_start = {0}".format(day_start))
        else:
            self.day_start = 0

        self.day = self.day_start
        self.data = self.df.loc[self.day,:]

        #initiate state
        self.state = self._initiate_state()                    #与self.data相关
        
        if self.initial:
            self.asset_memory = [self.initial_amount]
            self.cash_memory  = [self.initial_amount]
        else:
            previous_total_asset = self.previous_state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory = [previous_total_asset]
            self.cash_memory  = [self.previous_state[0]]

        self.turbulence = 0                             
        self.cost = 0                                     
        self.trades = 0                                 
        self.terminal = False                               
        # self.iteration=self.iteration
        self.rewards_memory = []                           
        self.actions_memory=[]                         
        self.date_memory=[self._get_date()]          
        
        self.episode+=1                                 

        return self.state                            
    
    def render(self, mode='human',close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:                              
            # For Initial State
            if len(self.df.tic.unique())>1:            
                # for multiple stock
                state = [self.initial_amount] + \
                         self.data.close.values.tolist() + \
                         [0]*self.stock_dim  + \
                         sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])   #flatten data

                if self.initial_buy:
                    state = self.initial_buy_()
            else:
                # for single stock
                state = [self.initial_amount] + \
                        [self.data.close] + \
                        [0]*self.stock_dim  + \
                        sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
        else:
            #Using Previous State
            if len(self.df.tic.unique())>1:
                # for multiple stock
                state = [self.previous_state[0]] + \
                         self.data.close.values.tolist() + \
                         self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]  + \
                         sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
            else:
                # for single stock
                state = [self.previous_state[0]] + \
                        [self.data.close] + \
                        self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]  + \
                        sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
        return state

    def _update_state(self):
        if len(self.df.tic.unique())>1:
            # for multiple stock
            state =  [self.state[0]] + \
                      self.data.close.values.tolist() + \
                      list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                      sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])


        else:
            # for single stock
            state =  [self.state[0]] + \
                     [self.data.close] + \
                     list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                     sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
                          
        return state

    def _get_date(self):
        if len(self.df.tic.unique())>1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_asset_memory(self):
        date_list = self.date_memory       
        asset_list = self.asset_memory    

        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique())>1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ['date']
            
            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


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
    
    def initial_buy_(self):
        """Initialize the state, already bought some"""
        prices = self.data.close.values.tolist()
        avg_price = sum(prices)/len(prices)
        buy_nums_each_tic = 0.5*self.initial_amount//(avg_price*len(prices))  # only use half of the initial amount
        cost = sum(prices)*buy_nums_each_tic

        state = [self.initial_amount-cost] + \
            self.data.close.values.tolist() + \
            [buy_nums_each_tic]*self.stock_dim  + \
            sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
        
        return state





