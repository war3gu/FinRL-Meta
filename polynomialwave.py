
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from datetime import datetime, timedelta


'''
def sin_wave(A, f, fs, phi, t):
    # 若时间序列长度为 t=1s,
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1/fs
    n = t / Ts
    n = np.arange(n)
    y = A*np.sin(2*np.pi*f*n*Ts + phi*(np.pi/180))
    return y


def generate_days(begin_date, count):
    date_list = []
    begin_date = datetime.strptime(begin_date, "%Y-%m-%d")
    for i in range(count):
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += timedelta(days=1)
    return np.array(date_list)


def generate_sin_wave(fs, last, noise_devia, stock_name):
    #noise = np.random.normal(0, noise_devia, last*fs)
    hz_50 = sin_wave(A=50, f=1, fs=fs, phi=0, t=last)
    hz_50_30 = sin_wave(A=20, f=2, fs=fs, phi=10, t=last)
    hz_50_60 = sin_wave(A=10, f=3, fs=fs, phi=160, t=last)
    hz_50_90 = sin_wave(A=4, f=4, fs=fs, phi=290, t=last)
    add = hz_50 + hz_50_30 + hz_50_60 + hz_50_90 #+ noise


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


    add = add - np.min(add) + 100   #避免股价为0，同时股价波动不能太变态

    date = generate_days('2000-01-01', fs*last)

    name = [stock_name] * (fs*last)

    name = np.array(name)

    #"tic"

    df = pd.DataFrame([add, date, name])
    df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    df.columns = ['close', 'date', 'tic']
    return df


fs1 = 20
last1 = 5
noise_devia1 = 3

df1 = generate_sin_wave(fs1, last1, noise_devia1, 'sinawave_noise1')
df1.to_csv('datasets/sinawave_noise1.csv')


fs2 = 25
last2 = 4
noise_devia2 = 10

df2 = generate_sin_wave(fs2, last2, noise_devia2, 'sinawave_noise2')
df2.to_csv('datasets/sinawave_noise2.csv')

'''

def generate_days(begin_date, count):
    date_list = []
    begin_date = datetime.strptime(begin_date, "%Y-%m-%d")
    for i in range(count):
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += timedelta(days=1)
    return np.array(date_list)

def f(coefficient, variable):
    p = 0
    for i in range(0,len(coefficient)):
        p = p + coefficient[i] * ((variable)**i)
    return p

function_coefficient1 = [0,-1,30, -30, -1, 1.5]

function_coefficient2 = [0,-1,20, -20, -1, 0.7]

data_length = 100
data_shift = -50

def generate_polynomial_wave(function_coefficient, data_length, data_shift, stock_name):
    data_array = []
    for i in range(data_length*2):
        x = i*0.5 + data_shift
        value = f(function_coefficient, x/10)
        data_array.append(value)

    print(data_array)
    data_array = (np.array(data_array)+1000)/100
    date = generate_days('2000-01-01', data_length*2)

    name = [stock_name] * (data_length*2)

    name = np.array(name)

    df = pd.DataFrame([data_array, date, name])
    df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    df.columns = ['close', 'date', 'tic']

    x = np.arange(0, data_length*2) + data_shift
    plt.xlabel('x')
    plt.ylabel('data1')
    plt.grid()
    plt.plot(x, data_array, 'k')
    plt.legend(['data1'], loc=1)
    plt.show()

    return df

df1 = generate_polynomial_wave(function_coefficient1, data_length, data_shift, 'polynomial_noise1.csv')
df1.to_csv('datasets/polynomial_noise1.csv')

df2 = generate_polynomial_wave(function_coefficient2, data_length, data_shift, 'polynomial_noise2.csv')
df2.to_csv('datasets/polynomial_noise2.csv')