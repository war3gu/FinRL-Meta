#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AI4Finance-Foundation/FinRL-Meta/blob/master/Demo_FinRL_Meta_Integrate_Trends_data_to_DOW_Jones.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


#get_ipython().run_cell_magic('capture', '', '!pip install git+https://github.com/AI4Finance-Foundation/FinRL.git')


# In[ ]:


#get_ipython().run_cell_magic('capture', '', '!pip3 install optuna')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

# matplotlib.use('Agg')
import datetime
import os
import optuna
import torch 

from finrl.apps import config
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
#from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline


# ## Custom data processor
# 
# * Only add a functionality add_user_defined_features to data processor 

# In[ ]:


from finrl.finrl_meta.data_processors.processor_alpaca import AlpacaProcessor as Alpaca
from finrl.finrl_meta.data_processors.processor_wrds import WrdsProcessor as Wrds
from finrl.finrl_meta.data_processors.processor_yahoofinance import YahooFinanceProcessor as YahooFinance
import pandas as pd
import numpy as np

class Custom_DataProcessor():
    def __init__(self, data_source, **kwargs):
        if data_source == 'alpaca':
            
            try:
                API_KEY= kwargs.get('API_KEY')
                API_SECRET= kwargs.get('API_SECRET')
                APCA_API_BASE_URL= kwargs.get('APCA_API_BASE_URL')
                self.processor = Alpaca(API_KEY, API_SECRET, APCA_API_BASE_URL)
                print('Alpaca successfully connected')
            except:
                raise ValueError('Please input correct account info for alpaca!')
                
        elif data_source == 'wrds':
            self.processor = Wrds()
            
        elif data_source == 'yahoofinance':
            self.processor = YahooFinance()
        
        else:
            raise ValueError('Data source input is NOT supported yet.')
    
    def download_data(self, ticker_list, start_date, end_date, 
                      time_interval):
        self.processor.download_data(ticker_list = ticker_list,
                                          start_date = start_date, 
                                          end_date = end_date,
                                          time_interval = time_interval)
        self.dataframe = self.processor.dataframe
    
    def clean_data(self):
        self.processor.clean_data()
        self.dataframe = self.processor.dataframe
    
    def add_technical_indicator(self, tech_indicator_list):
        # self.tech_indicator_list = tech_indicator_list
        self.processor.add_technical_indicator(tech_indicator_list)
        self.dataframe = self.processor.dataframe
    
    def add_turbulence(self):
        self.processor.add_turbulence(df)
        self.dataframe = self.processor.dataframe
    
    def add_vix(self):
        self.processor.add_vix()
        self.dataframe = self.processor.dataframe
    
    def add_user_defined_features(self,user_df):
        df = self.processor.dataframe.copy()
        df = df.merge(user_df, how='left', left_on=[
            'time', 'tic'], right_on=['time', 'tic'])
        self.processor.dataframe = df
        self.dataframe = df
    
    def df_to_array(self, tech_indicator_list, if_vix) -> np.array:
        price_array,tech_array,turbulence_array = self.processor.df_to_array(
                                                tech_indicator_list,
                                                if_vix)
        #fill nan with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0
        
        return price_array,tech_array,turbulence_array


# ## User defined column

# In[ ]:


# !gdown --id "1sp11dtAJGGqC-3UdSn774ZD1zWCsqbn4"
#get_ipython().system('gdown --id "1m63ncE-BYlS77u5ejYTte9Nmh35DWhzp"')


# In[ ]:


#get_ipython().system('unzip "/content/Pytrends.zip"')


# In[ ]:


ticker_list = config.DOW_30_TICKER
#Pytrends dataframe
def get_user_df():
    pytrends_list = os.listdir('Pytrends_Data')
    
    user_df = pd.DataFrame()
    for pytrend in pytrends_list:
        tic_name = pytrend.split('_')[0]
        if tic_name in ticker_list:
            file_name = os.path.join('Pytrends_Data', pytrend)
            temp_user_df = pd.read_csv(file_name)
            temp_user_df.rename(columns={temp_user_df.columns[1]:'trends'},inplace=True)
            temp_user_df.rename(columns={temp_user_df.columns[0]:'time'},inplace=True)
            temp_user_df['tic'] = tic_name
            user_df = user_df.append(temp_user_df, ignore_index=True)
    return user_df


# In[ ]:


user_df = get_user_df()
len(user_df)


# ## Training and testing

# In[ ]:


technical_indicator_list = []

info_col = technical_indicator_list + ['trends']


# In[ ]:


def ppo_sample_parameters(trial:optuna.Trial):
  batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
  n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
  learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
  ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)

  return {
      "batch_size": batch_size,
      'n_steps': n_steps,
      'learning_rate': learning_rate,
      'ent_coef': ent_coef 
  }


# In[ ]:


os.makedirs('tuned_models',exist_ok=True)


# In[ ]:


from IPython.display import clear_output
def get_train_env(start_date, end_date, ticker_list, data_source, time_interval,model_name,env,
          info_col, if_vix=True,
          **kwargs):

    DP = Custom_DataProcessor(data_source, **kwargs)
    DP.download_data(ticker_list, start_date, end_date, time_interval)
    DP.clean_data()
    DP.add_user_defined_features(user_df) #Adding Google trends data to our state space
    DP.add_technical_indicator(technical_indicator_list)
    if if_vix:
        DP.add_vix()
    # Passed info col instead of tech_indicator_list.
    price_array, tech_array, turbulence_array = DP.df_to_array(
        data,info_col, if_vix)
    env_config = {'price_array':price_array,
              'tech_array':tech_array,
              'turbulence_array':turbulence_array,
              'if_train':True}
    env_instance = env(config=env_config)
    

    return env_instance

def objective(trial:optuna.Trial):
    agent_params = ppo_sample_parameters(trial)
    tune_cwd = 'tuned_models/'+str(model_name)+'_' + str(agent_params.values())
    agent = DRLAgent_sb3(env = train_env_instance)

    model = agent.get_model(model_name, model_kwargs = agent_params)
    trained_model = agent.train_model(model=model, 
                            tb_log_name=model_name,
                            total_timesteps=total_timesteps)
    clear_output(wait=True)
    trained_model.save(tune_cwd)

    val_sharpe,_ = val_or_test(val_env_instance,tune_cwd,model_name)

    return val_sharpe


# In[ ]:


def calculate_sharpe(df):
  df['daily_return'] = df['account_value'].pct_change(1)
  if df['daily_return'].std() !=0:
    sharpe = (252**0.5)*df['daily_return'].mean()/           df['daily_return'].std()
    return sharpe
  else:
    return 0


# In[ ]:


def get_test_env(start_date, end_date, ticker_list, data_source, time_interval, 
         info_col, env, model_name, if_vix = True,
         **kwargs):
    #fetch data
    DP = Custom_DataProcessor(data_source, **kwargs)
    DP.download_data(ticker_list, start_date, end_date, time_interval)
    DP.clean_data()
    DP.add_user_defined_features(user_df)
    DP.add_technical_indicator(technical_indicator_list)
    
    if if_vix:
        DP.add_vix()
    price_array, tech_array, turbulence_array = DP.df_to_array(info_col, if_vix)
    
    env_config = {'price_array':price_array,
            'tech_array':tech_array,
            'turbulence_array':turbulence_array,
            'if_train':False}
    test_env_instance = env(config=env_config)
    return test_env_instance

def val_or_test(test_env_instance,cwd,model_name): 
    episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
                                  model_name=model_name, 
                                  environment = test_env_instance,
                                  cwd = cwd)
    sharpe_df = pd.DataFrame(episode_total_assets,columns=['account_value'])

    return calculate_sharpe(sharpe_df),sharpe_df


# In[ ]:


from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv


# In[ ]:


TRAIN_START_DATE = '2012-01-01'
TRAIN_END_DATE = '2019-07-30'

VAL_START_DATE = '2019-08-01'
VAL_END_DATE = '2020-07-30'
TEST_START_DATE = '2020-08-01'
TEST_END_DATE = '2021-10-01'

PPO_PARAMS = {"n_steps": 2048,"ent_coef": 0.01,"learning_rate": 0.00025,"batch_size": 128}
SAC_PARAMS = {"batch_size": 128,"buffer_size": 100000,"learning_rate": 0.0001,"learning_starts": 100,"ent_coef": "auto_0.1",}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 3e-5}


data_source = 'yahoofinance'
time_interval = '1D'
model_name = 'ppo'
total_timesteps = 30000

env = StockTradingEnv


# In[ ]:


train_env_instance = get_train_env(TRAIN_START_DATE, TRAIN_END_DATE, 
                                   ticker_list, data_source, 
                                   time_interval,model_name,
                                   env,info_col)
val_env_instance = get_test_env(VAL_START_DATE, VAL_END_DATE, 
                                ticker_list, data_source,
                                time_interval, info_col, env, model_name)                         


# In[ ]:


sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(study_name="ppo_study",direction='maximize',
                            sampler = sampler, pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=30,catch=(ValueError,))


# In[ ]:


print(study.best_params)
print(study.best_value)
print(study.best_trial)


# In[ ]:


test_env_instance = get_test_env(TEST_START_DATE, TEST_END_DATE, 
                                ticker_list, data_source,
                                time_interval, info_col, env, model_name)    
test_cwd = 'tuned_models/'+str(model_name)+'_' + str(study.best_params.values())
test_sharpe,df_account_value = val_or_test(test_env_instance,test_cwd,model_name)


# ## Backtesting

# In[ ]:


Custom_DataProcessor('yahoofinance').download_data(ticker_list = ["^DJI"],
                                                            start_date = TEST_START_DATE, 
                                                            end_date = TEST_END_DATE, 
                                                            time_interval = "1D")
stats = backtest_stats(Custom_DataProcessor.dataframe, value_col_name = 'close')


# In[ ]:


os.chdir('/content/tuned_models')
for test_cwd in os.listdir():
  test_sharpe,df_account_value = val_or_test(test_env_instance,test_cwd,model_name)
  print(test_cwd,test_sharpe)


# In[ ]:


# !mkdir 'drive/MyDrive/tuned_models_DOW_JONES'
# !mv 'tuned_models_DOW_JONES' 'drive/MyDrive/tuned_models'


# In[ ]:


#Best test sharpe
best_test_cwd = "ppo_dict_values([64, 128, 0.0007114879943759374, 1.7734195965746112e-05])"
# ppo_dict_values([256, 256, 0.00013931273790066692, 3.4582737549732e-08])

test_sharpe,df_account_value = val_or_test(test_env_instance,best_test_cwd,model_name)


# In[ ]:


account_value_sb3 = list(df_account_value['account_value'])
account_value_sb3 = pd.DataFrame({'date':Custom_DataProcessor.dataframe.date,'account_value':account_value_sb3[0:len(account_value_sb3)-1]})
perf_stats_all = backtest_stats(account_value=account_value_sb3)
perf_stats_all = pd.DataFrame(perf_stats_all)


# In[ ]:


account_value_sb3.tail()


# In[ ]:


print("==============Compare to DJIA===========")
get_ipython().run_line_magic('matplotlib', 'inline')
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX

backtest_plot(account_value_sb3, 
             baseline_ticker = '^DJI', 
             baseline_start = account_value_sb3.loc[0,'date'],
             baseline_end = account_value_sb3.loc[len(account_value_sb3)-1,'date'])


# In[ ]:




