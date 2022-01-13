#%%
from train import train
from test import test
from finrl_meta.env_crypto_trading.env_multiple_crypto import CryptoEnv
import numpy as np
import pandas as pd

print("imports done")
# Default Parameters
# TICKER_LIST = ['BTCUSDT','ETHUSDT','ADAUSDT','BNBUSDT','XRPUSDT','SOLUSDT','DOTUSDT',
#          'DOGEUSDT','AVAXUSDT','UNIUSDT']

#exchanged avax,uni with LTC, and BCH
TICKER_LIST = ['BTCUSDT','ETHUSDT','ADAUSDT','BNBUSDT','XRPUSDT','DOTUSDT',
        'DOGEUSDT','LTCUSDT','BCHUSDT']
env = CryptoEnv

# Train for 20 days, testing 10 days (4564 * 10)
TRAIN_START_DATE = '2021-09-02'
TRAIN_END_DATE = '2021-09-21'
TEST_START_DATE = '2021-09-21'
TEST_END_DATE = '2021-09-30'

#Train for 1 year, test on September (data for dates not equal for all securities)
# using new (LTC, BCH) for this
# TRAIN_START_DATE = '2020-09-01'
# TRAIN_END_DATE = '2021-08-31'
# TEST_START_DATE = '2021-09-01'
# TEST_END_DATE = '2021-09-30'

# Train for 8 mos, test on September
# TRAIN_START_DATE = '2021-01-01'
# TRAIN_END_DATE = '2021-08-30'
# TEST_START_DATE = '2021-09-01'
# TEST_END_DATE = '2021-09-30'

# Train for 2 mos, test on September
# TRAIN_START_DATE = '2021-07-01'
# TRAIN_END_DATE = '2021-08-30'
# TEST_START_DATE = '2021-09-01'
# TEST_END_DATE = '2021-09-30'

# Train for 1 mo, test on September
# TRAIN_START_DATE = '2021-08-01'
# TRAIN_END_DATE = '2021-08-30'
# TEST_START_DATE = '2021-09-01'
# TEST_END_DATE = '2021-09-30'

# Train for 40 days, test on last 10 days of September
# TRAIN_START_DATE = '2021-08-01'
# TRAIN_END_DATE = '2021-09-20'
# TEST_START_DATE = '2021-09-21'
# TEST_END_DATE = '2021-09-30'


TECHNICAL_INDICATORS_LIST = ['macd','cci','dx'] #self-defined technical indicator list is NOT supported yet

#target step controls ...XYZ
TARGET_STEP = 100000 #500000
ERL_PARAMS = {"learning_rate": 2 ** -15,"batch_size": 2**11,"gamma":  0.99,
              "seed":312,"net_dimension": 2**9, "target_step": TARGET_STEP, 
              "eval_time_gap": 30}

DATA_TIME_INTERVAL = '5m'
# training breaks when agent conducts BREAK_STEP steps
BREAK_STEP = 1e5 #2e5
#%%
#Training:
train(start_date = TRAIN_START_DATE, 
      end_date = TRAIN_END_DATE,
      ticker_list = TICKER_LIST, 
      data_source = 'binance',
      time_interval= DATA_TIME_INTERVAL, 
      technical_indicator_list= TECHNICAL_INDICATORS_LIST,
      drl_lib='elegantrl', 
      env=env, 
      model_name='ppo', 
      current_working_dir='./test_ppo',
      erl_params=ERL_PARAMS,
      break_step=BREAK_STEP,
      if_vix=False,
      customf="custom2.csv",
      )

#%%
# #Testing
account_value_erl = test(start_date = TEST_START_DATE, 
                        end_date = TEST_END_DATE,
                        ticker_list = TICKER_LIST, 
                        data_source = 'binance',
                        time_interval= '5m', 
                        technical_indicator_list= TECHNICAL_INDICATORS_LIST,
                        drl_lib='elegantrl', 
                        env=env, 
                        model_name='ppo', 
                        current_working_dir='./test_ppo', 
                        net_dimension = 2**9)

#%%
#Plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

#calculate agent returns
account_value_erl = np.array(account_value_erl)
agent_returns = account_value_erl/account_value_erl[0]
#calculate buy-and-hold btc returns
price_array = np.load('./price_array.npy')
btc_prices = price_array[:,0]
buy_hold_btc_returns = btc_prices/btc_prices[0]
#calculate equal weight portfolio returns
price_array = np.load('./price_array.npy')
initial_prices = price_array[0,:]
equal_weight = np.array([1e5/initial_prices[i] for i in range(10)])
equal_weight_values = []
for i in range(0, price_array.shape[0]):
  equal_weight_values.append(np.sum(equal_weight * price_array[i]))
equal_weight_values = np.array(equal_weight_values)
equal_returns = equal_weight_values/equal_weight_values[0]

modelparams = {
    "trStart": TRAIN_START_DATE, 
    "trEnd": TRAIN_END_DATE, 
    "teStart": TEST_START_DATE, 
    "teEnd": TRAIN_END_DATE, 
    "timeInt": DATA_TIME_INTERVAL, 
    "tarStep": TARGET_STEP,
    "brStep": BREAK_STEP,
    } 
from datetime import datetime as dt

def img_naming(params) -> str:
    pathname = "tr_"+params['trStart']+"_to_"+params['trEnd']+"_with_interval_"+\
        params['timeInt']+"_brStep_"+str(params['brStep'])+"_tarStep_"+str(params['tarStep'])+\
            "_testedOn_"+params['teStart']+"_to_"+params['teEnd']+".png"
    return pathname
    #tr_days = str((dt.strptime(params['trEnd'], "%Y-%m-%d") - dt.strptime(params['trStart'], "%Y-%m-%d")).days)

#plot 
#%%
plt.figure()
plt.grid()
plt.grid(which='minor', axis='y')
plt.title('Cryptocurrency Trading ', fontsize=20)
plt.plot(agent_returns, label = 'ElegantRL Agent', color = 'red')
plt.plot(buy_hold_btc_returns, label = 'Buy-and-Hold BTC', color = 'blue')
plt.plot(equal_returns, label = 'Equal Weight Portfolio', color = 'green')
plt.ylabel('Return', fontsize=16)
plt.xlabel('Times (5min)', fontsize=16)
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.legend(fontsize=10.5)
plt.show()
plt.savefig(img_naming(modelparams))

#%%
# TICKER_LIST.insert(0,"CASH")
# print(TICKER_LIST)
# print(footprint.shape)
# fp = pd.DataFrame(data=footprint, columns=TICKER_LIST)

#%%
# print(fp.head())

# %%
#Make stacked area plot
import seaborn as sns
plt.rcParams['figure.figsize'] = [20, 10]
#clip 0s in dataframe...shouldnt have zeros but we do...
# fp[fp < 0] = 0
# fp.plot.area()
plt.show(block=False)
# %%
