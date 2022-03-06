
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from datetime import datetime, timedelta

def sin_wave(A, f, fs, phi, t):
    '''
    :params A:    振幅
    :params f:    信号频率       1秒多少个波峰
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    '''
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
    noise = np.random.normal(0, noise_devia, last*fs)
    hz_50 = sin_wave(A=50, f=1, fs=fs, phi=0, t=last)
    hz_50_30 = sin_wave(A=20, f=2, fs=fs, phi=10, t=last)
    hz_50_60 = sin_wave(A=10, f=3, fs=fs, phi=160, t=last)
    hz_50_90 = sin_wave(A=4, f=4, fs=fs, phi=290, t=last)
    add = hz_50 + hz_50_30 + hz_50_60 + hz_50_90 + noise
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
    add = add - np.min(add) + 1   #避免价格为0

    date = generate_days('2000-01-01', fs*last)

    name = [stock_name] * (fs*last)

    name = np.array(name)

    #"tic"

    df = pd.DataFrame([add, date, name])
    df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    df.columns = ['close', 'date', 'tic']
    return df


fs1 = 100
last1 = 30
noise_devia1 = 3

df1 = generate_sin_wave(fs1, last1, noise_devia1, 'sinawave_noise1')
df1.to_csv('datasets/sinawave_noise1.csv')


fs2 = 150
last2 = 20
noise_devia2 = 10

df2 = generate_sin_wave(fs2, last2, noise_devia2, 'sinawave_noise2')
df2.to_csv('datasets/sinawave_noise2.csv')


