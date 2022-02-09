#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AI4Finance-Foundation/FinRL-Meta/blob/master/Demo_Plug_and_Play_with_DRL_Libraries.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # FinRL-Meta: Plug-and-Play with DRL Libraries
# Demostration for plug-and-play with ElegantRL, Stable-baselines3, RLlib

# ## Part 1: Getting Started - Install Python Packages 

#  ### 1.1 Install DRL libraries: FinRL, ElegantRL, RLlib

# In[ ]:


## install elegantrl library
get_ipython().system('pip install git+https://github.com/AI4Finance-LLC/ElegantRL.git')
## install rllib/ray library
get_ipython().system('pip install ray[default]')
## install finrl library
get_ipython().system('pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git')


# ### 1.2 Check if the additional packages needed are present, if not install them

# In[ ]:


# !pip install trading_calendars
# !pip install alpaca_trade_api
# !pip install ccxt
# !pip install jqdatasdk
# !pip install wrds

# !pip install lz4
# !pip install ray[tune]
# !pip install tensorboardX
# !pip install gputil


# ### 1.3 Import packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# matplotlib.use('Agg')
import datetime

import torch 
import ray
from finrl.apps import config
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
#from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from finrl.drl_agents.rllib.models import DRLAgent as DRLAgent_rllib
from finrl.drl_agents.elegantrl.models import DRLAgent as DRLAgent_erl

from finrl.finrl_meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline


# ## Part 2: Train & Test Function

# ### 2.1 Train

# In[ ]:


def train(start_date, end_date, ticker_list, data_source, time_interval, 
          technical_indicator_list, drl_lib, env, model_name, if_vix = True,
          **kwargs):
    
    #fetch data
    DP = DataProcessor(data_source, **kwargs)
    DP.download_data(ticker_list, start_date, end_date, time_interval)
    DP.clean_data()
    DP.add_technical_indicator(technical_indicator_list)
    if if_vix:
        DP.add_vix()
    price_array, tech_array, turbulence_array = DP.df_to_array(if_vix)
    env_config = {'price_array':price_array,
              'tech_array':tech_array,
              'turbulence_array':turbulence_array,
              'if_train':True}
    env_instance = env(config=env_config)

    #read parameters
    cwd = kwargs.get('cwd','./'+str(model_name))

    if drl_lib == 'elegantrl':
        break_step = kwargs.get('break_step', 1e6)
        erl_params = kwargs.get('erl_params')

        agent = DRLAgent_erl(env = env,
                             price_array = price_array,
                             tech_array=tech_array,
                             turbulence_array=turbulence_array)
        
        model = agent.get_model(model_name, model_kwargs = erl_params)
        trained_model = agent.train_model(model=model, 
                                          cwd=cwd,
                                          total_timesteps=break_step)
      
    elif drl_lib == 'rllib':
        total_episodes = kwargs.get('total_episodes', 100)
        rllib_params = kwargs.get('rllib_params')

        agent_rllib = DRLAgent_rllib(env = env,
                       price_array=price_array,
                       tech_array=tech_array,
                       turbulence_array=turbulence_array)

        model,model_config = agent_rllib.get_model(model_name)

        model_config['lr'] = rllib_params['lr']
        model_config['train_batch_size'] = rllib_params['train_batch_size']
        model_config['gamma'] = rllib_params['gamma']

        #ray.shutdown()
        trained_model = agent_rllib.train_model(model=model, 
                                          model_name=model_name,
                                          model_config=model_config,
                                          total_episodes=total_episodes)
        trained_model.save(cwd)
        
            
    elif drl_lib == 'stable_baselines3':
        total_timesteps = kwargs.get('total_timesteps', 1e6)
        agent_params = kwargs.get('agent_params')

        agent = DRLAgent_sb3(env = env_instance)

        model = agent.get_model(model_name, model_kwargs = agent_params)
        trained_model = agent.train_model(model=model, 
                                tb_log_name=model_name,
                                total_timesteps=total_timesteps)
        print('Training finished!')
        trained_model.save(cwd)
        print('Trained model saved in ' + str(cwd))
    else:
        raise ValueError('DRL library input is NOT supported. Please check.')


# ### 2.2 Test

# In[ ]:


def test(start_date, end_date, ticker_list, data_source, time_interval, 
         technical_indicator_list, drl_lib, env, model_name, if_vix = True,
         **kwargs):
    #fetch data
    DP = DataProcessor(data_source, **kwargs)
    DP.download_data(ticker_list, start_date, end_date, time_interval)
    DP.clean_data()
    DP.add_technical_indicator(technical_indicator_list)
    
    if if_vix:
        DP.add_vix()
    price_array, tech_array, turbulence_array = DP.df_to_array(if_vix)
    
    env_config = {'price_array':price_array,
            'tech_array':tech_array,
            'turbulence_array':turbulence_array,
            'if_train':False}
    env_instance = env(config=env_config)

    #load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get('net_dimension', 2**7)
    cwd = kwargs.get('cwd','./'+str(model_name))
    print("price_array: ",len(price_array))

    if drl_lib == 'elegantrl':
        episode_total_assets = DRLAgent_erl.DRL_prediction(model_name=model_name,
                                            cwd=cwd,
                                            net_dimension=net_dimension,
                                            environment=env_instance)

        return episode_total_assets
    
    elif drl_lib == 'rllib':
        #load agent
        episode_total_assets = DRLAgent_rllib.DRL_prediction(
                                  model_name=model_name, 
                                  env = env,
                                  price_array=price_array,
                                  tech_array=tech_array,
                                  turbulence_array=turbulence_array,
                                  agent_path = cwd)

        return episode_total_assets


    elif drl_lib == 'stable_baselines3':
        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
                                  model_name=model_name, 
                                  environment = env_instance,
                                  cwd = cwd)
        
        return episode_total_assets
    else:
        raise ValueError('DRL library input is NOT supported. Please check.')


# ## Part 3: Set DRL Environment

# ### 3.1 Get the stock trading env from neo_finrl

# In[ ]:


from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv


# In[ ]:


import numpy as np
import os
import gym
from numpy import random as rd

class StockTradingEnv(gym.Env):

    def __init__(self, config, initial_account=1e6,
                 gamma=0.99, turbulence_thresh=99, min_stock_rate=0.1,
                 max_stock=1e2, initial_capital=1e6, buy_cost_pct=1e-3, 
                 sell_cost_pct=1e-3,reward_scaling=2 ** -11,  initial_stocks=None,
                 ):
        price_ary = config['price_array']
        tech_ary = config['tech_array']
        turbulence_ary = config['turbulence_array']
        if_train = config['if_train']
        n = price_ary.shape[0]
        self.price_ary =  price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary
        
        self.tech_ary = self.tech_ary * 2 ** -7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2 ** -5).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = np.zeros(stock_dim, dtype=np.float32) if initial_stocks is None else initial_stocks

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = 'StockEnv'
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 5.0
        self.episode_return = 0.0
        
        self.observation_space = gym.spaces.Box(low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        
    def reset(self):
        self.day = 0
        price = self.price_ary[self.day]
        
        if self.if_train:
            self.stocks = (self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)).astype(np.float32)
            self.stocks_cd = np.zeros_like(self.stocks)
            self.amount = self.initial_capital * rd.uniform(0.95, 1.05) - (self.stocks * price).sum()
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cd = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price)  # state

    def step(self, actions):
        actions = (actions * self.max_stock).astype(int)

        self.day += 1
        price = self.price_ary[self.day]
        self.stocks_cd += 1

        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    self.stocks[index] -= sell_num_shares
                    self.amount += price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    self.stocks_cd[index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if price[index] > 0:  # Buy only if the price is > 0 (no missing data in this particular date)
                    buy_num_shares = min(self.amount // price[index], actions[index])
                    self.stocks[index] += buy_num_shares
                    self.amount -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                    self.stocks_cd[index] = 0

        else:  # sell all when turbulence
            self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cd[:] = 0

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, dict()

    def get_state(self, price):
        amount = np.array(max(self.amount, 1e4) * (2 ** -12), dtype=np.float32)
        scale = np.array(2 ** -6, dtype=np.float32)
        return np.hstack((amount,
                          self.turbulence_ary[self.day],
                          self.turbulence_bool[self.day],
                          price * scale,
                          self.stocks * scale,
                          self.stocks_cd,
                          self.tech_ary[self.day],
                          ))  # state.astype(np.float32)
    
    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh


# ### 3.2 Set some basic parameters

# In[ ]:


env = StockTradingEnv


# In[ ]:


TRAIN_START_DATE = '2014-01-01'
TRAIN_END_DATE = '2020-07-30'

TEST_START_DATE = '2020-08-01'
TEST_END_DATE = '2021-10-01'
TECHNICAL_INDICATORS_LIST = ['macd',
 'boll_ub',
 'boll_lb',
 'rsi_30',
 'dx_30',
 'close_30_sma',
 'close_60_sma']


# ## Part 4: Compare the three agents

# ### 4.1 eRL

# In[ ]:


ERL_PARAMS = {"learning_rate": 3e-5,"batch_size": 2048,"gamma":  0.985,
        "seed":312,"net_dimension":512, "target_step":5000, "eval_gap":60,
        "eval_times":1}


# #### Train

# In[ ]:


#demo for elegantrl
train(start_date = TRAIN_START_DATE, 
      end_date = TRAIN_END_DATE,
      ticker_list = config.DOW_30_TICKER, 
      data_source = 'yahoofinance',
      time_interval= '1D', 
      technical_indicator_list= TECHNICAL_INDICATORS_LIST,
      drl_lib='elegantrl', 
      env=env, 
      model_name='ppo', 
      cwd='./test_ppo',
      erl_params=ERL_PARAMS,
      break_step=1e5
      )


# #### Test

# In[ ]:


account_value_erl=test(start_date = TEST_START_DATE, 
                        end_date = TEST_END_DATE,
                        ticker_list = config.DOW_30_TICKER, 
                        data_source = 'yahoofinance',
                        time_interval= '1D', 
                        technical_indicator_list= TECHNICAL_INDICATORS_LIST,
                        drl_lib='elegantrl', 
                        env=env, 
                        model_name='ppo', 
                        cwd='./test_ppo', 
                        net_dimension = 512)


# In[ ]:


len(account_value_erl)


# #### Plot

# In[ ]:


TEST_END_DATE


# In[ ]:


DP = DataProcessor('yahoofinance')
DP.download_data(ticker_list = ["^DJI"],
                start_date = TEST_START_DATE,
                end_date = TEST_END_DATE,
                time_interval = "1D")
stats = backtest_stats(DP.dataframe, value_col_name = 'close')


# In[ ]:


account_value_erl = pd.DataFrame({'date':DP.dataframe.date,'account_value':account_value_erl[0:len(account_value_erl)-1]})


# In[ ]:


account_value_erl.tail()


# In[ ]:


print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=account_value_erl)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+"/perf_stats_all_"+now+'.csv')


# In[ ]:





# In[ ]:


print("==============Compare to DJIA===========")
get_ipython().run_line_magic('matplotlib', 'inline')
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(account_value_erl, 
             baseline_ticker = '^DJI', 
             baseline_start = account_value_erl.loc[0,'date'],
             baseline_end = account_value_erl.loc[len(account_value_erl)-1,'date'])


# In[ ]:





# ### 4.2 RLlib

# In[ ]:


RLlib_PARAMS = {"lr": 5e-6,"train_batch_size": 1000,"gamma": 0.99}


# #### Train

# In[ ]:


#demo for rllib
ray.shutdown() #always shutdown previous session if any

train(start_date = TRAIN_START_DATE, 
      end_date = TRAIN_END_DATE,
      ticker_list = config.DOW_30_TICKER, 
      data_source = 'yahoofinance',
      time_interval= '1D', 
      technical_indicator_list= TECHNICAL_INDICATORS_LIST,
      drl_lib='rllib', 
      env=env, 
      model_name='ppo', 
      cwd='./test_ppo',
      rllib_params = RLlib_PARAMS,
      total_episodes=30)


# #### Test

# In[ ]:


ray.shutdown() #always shutdown previous session if any

account_value_rllib = test(start_date = TEST_START_DATE, 
     end_date = TEST_END_DATE,
     ticker_list = config.DOW_30_TICKER, 
     data_source = 'yahoofinance',
     time_interval= '1D', 
     technical_indicator_list= TECHNICAL_INDICATORS_LIST,
     drl_lib='rllib', 
     env=env, 
     model_name='ppo', 
     cwd='./test_ppo/checkpoint_000030/checkpoint-30',
     rllib_params = RLlib_PARAMS)


# In[ ]:


len(account_value_rllib)


# #### Plot

# In[ ]:


DP = DataProcessor('yahoofinance')
DP.download_data(ticker_list = ["^DJI"],
                start_date = TEST_START_DATE,
                end_date = TEST_END_DATE,
                time_interval = "1D")
stats = backtest_stats(DP.dataframe, value_col_name = 'close')


# In[ ]:


len(DP.dataframe.date)


# In[ ]:


account_value_rllib = pd.DataFrame({'date':DP.dataframe.date,'account_value':account_value_rllib[0:len(account_value_rllib)-1]})


# In[ ]:


perf_stats_all = backtest_stats(account_value=account_value_rllib)
perf_stats_all = pd.DataFrame(perf_stats_all)


# In[ ]:


print("==============Compare to DJIA===========")
get_ipython().run_line_magic('matplotlib', 'inline')
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(account_value_rllib, 
             baseline_ticker = '^DJI', 
             baseline_start = account_value_rllib.loc[0,'date'],
             baseline_end = account_value_rllib.loc[len(account_value_rllib)-1,'date'])


# ### 4.3 Stable-baselines3

# In[ ]:


SAC_PARAMS = {"batch_size": 128,"buffer_size": 100000,"learning_rate": 0.0001,"learning_starts": 100,"ent_coef": "auto_0.1",}
PPO_PARAMS = {"n_steps": 2048,"ent_coef": 0.01,"learning_rate": 0.00025,"batch_size": 128}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 3e-5}


# #### Train

# In[ ]:


#demo for stable-baselines3
train(start_date = TRAIN_START_DATE, 
      end_date = TRAIN_END_DATE,
      ticker_list = config.DOW_30_TICKER, 
      data_source = 'yahoofinance',
      time_interval= '1D', 
      technical_indicator_list= TECHNICAL_INDICATORS_LIST,
      drl_lib='stable_baselines3', 
      env=env, 
      model_name='sac', 
      cwd='./test_sac',
      agent_params = SAC_PARAMS,
      total_timesteps=1e4)


# #### Test

# In[ ]:


account_value_sb3=test(start_date = TEST_START_DATE, 
     end_date = TEST_END_DATE,
     ticker_list = config.DOW_30_TICKER, 
     data_source = 'yahoofinance',
     time_interval= '1D', 
     technical_indicator_list= TECHNICAL_INDICATORS_LIST, 
     drl_lib='stable_baselines3', 
     env=env, 
     model_name='sac', 
     cwd='./test_sac.zip')


# In[ ]:


len(account_value_sb3)


# #### Plot

# In[ ]:


DP = DataProcessor('yahoofinance')
DP.download_data(ticker_list = ["^DJI"],
                start_date = TEST_START_DATE,
                end_date = TEST_END_DATE,
                time_interval = "1D")
stats = backtest_stats(DP.dataframe, value_col_name = 'close')


# In[ ]:


account_value_sb3 = pd.DataFrame({'date':DP.dataframe.date,'account_value':account_value_sb3[0:len(account_value_sb3)-1]})


# In[ ]:


perf_stats_all = backtest_stats(account_value=account_value_sb3)
perf_stats_all = pd.DataFrame(perf_stats_all)


# In[ ]:


account_value_sb3.tail()


# In[ ]:


print("==============Compare to DJIA===========")
#%matplotlib inline
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(account_value_sb3, 
             baseline_ticker = '^DJI', 
             baseline_start = account_value_sb3.loc[0,'date'],
             baseline_end = account_value_sb3.loc[len(account_value_sb3)-1,'date'])


# ## Part 5: Use Plotly to compare eRL, RLlib and SB3

# In[ ]:


DP.dataframe


# In[ ]:


from datetime import datetime as dt

import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go


# In[ ]:


daily_return = account_value_sb3.copy()
daily_return['sb3_return'] = account_value_sb3.account_value.pct_change()
daily_return['erl_return'] = account_value_erl.account_value.pct_change()
daily_return['rllib_return'] = account_value_rllib.account_value.pct_change()
daily_return['djia_return'] = DP.dataframe.adjcp.pct_change()


# In[ ]:


daily_return.head()


# In[ ]:


daily_return.to_csv('daily_return_erl_sb3_rllib.csv',index=False)
#daily_return = pd.read_csv('daily_return_erl_sb3_rllib.csv')


# In[ ]:


rllib_cumpod =(daily_return.rllib_return+1).cumprod()-1
sb3_cumpod =(daily_return.sb3_return+1).cumprod()-1
erl_cumpod =(daily_return.erl_return+1).cumprod()-1
dji_cumpod =(daily_return.djia_return+1).cumprod()-1


# In[ ]:





# In[ ]:


time_ind = pd.Series(daily_return.date)


# In[ ]:


trace0_portfolio = go.Scatter(x = time_ind, y = rllib_cumpod, mode = 'lines', name = 'RLlib')

trace1_portfolio = go.Scatter(x = time_ind, y = dji_cumpod, mode = 'lines', name = 'DJIA')
trace2_portfolio = go.Scatter(x = time_ind, y = sb3_cumpod, mode = 'lines', name = 'Stablebaselines3')
trace3_portfolio = go.Scatter(x = time_ind, y = erl_cumpod, mode = 'lines', name = 'ElegantRL')
#trace4_portfolio = go.Scatter(x = time_ind, y = addpg_cumpod, mode = 'lines', name = 'Adaptive-DDPG')
#trace5_portfolio = go.Scatter(x = time_ind, y = min_cumpod, mode = 'lines', name = 'Min-Variance')

#trace4 = go.Scatter(x = time_ind, y = addpg_cumpod, mode = 'lines', name = 'Adaptive-DDPG')

#trace2 = go.Scatter(x = time_ind, y = portfolio_cost_minv, mode = 'lines', name = 'Min-Variance')
#trace3 = go.Scatter(x = time_ind, y = spx_value, mode = 'lines', name = 'SPX')


# In[ ]:


fig = go.Figure()
fig.add_trace(trace3_portfolio)
fig.add_trace(trace2_portfolio)

fig.add_trace(trace0_portfolio)
fig.add_trace(trace1_portfolio)




fig.update_layout(
    legend=dict(
        x=0,
        y=1,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=15,
            color="black"
        ),
        bgcolor="White",
        bordercolor="white",
        borderwidth=2
        
    ),
)
#fig.update_layout(legend_orientation="h")
fig.update_layout(title={
        #'text': "Cumulative Return using FinRL",
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
#with Transaction cost
#fig.update_layout(title =  'Quarterly Trade Date')
fig.update_layout(
#    margin=dict(l=20, r=20, t=20, b=20),

    paper_bgcolor='rgba(1,1,0,0)',
    plot_bgcolor='rgba(1, 1, 0, 0)',
    #xaxis_title="Date",
    yaxis_title="Cumulative Return",
xaxis={'type': 'date', 
       'tick0': time_ind[0], 
        'tickmode': 'linear', 
       'dtick': 86400000.0 *70}

)
fig.update_xaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
fig.update_yaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')

fig.show()


# In[ ]:




