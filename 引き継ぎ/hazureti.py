import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("aweek.csv", usecols=[0,1])
print(df.head())

# time
#x = df['timestamp'].values

# day
x = df['DAY'].values

y = df['RESULT'].values


print('x:',x)
print('y:',y)

#plt.plot(x,y)
#plt.show()


# span が計算する期間
ewm_mean = df.ewm(span=16).mean()  # 指数加重移動平均

fig, ax = plt.subplots()
ax.plot(y, label='original')
ax.plot(ewm_mean, label='ema')
plt.xlabel("Date", fontsize=18)
plt.ylabel("10^4kW", fontsize=18)
#ax.legend()
#plt.show()


# threshold で何倍以上を外れ値として扱うかを示す
def plot_outlier(y, ewm_span=16, threshold=1.0):
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.rcParams["font.size"] = 15
    y = pd.Series(y)
    fig, ax = plt.subplots()

    # 指数加重移動平均
    ewm_mean = y.ewm(span=ewm_span).mean()
    # 指数加重移動標準偏差
    ewm_std = y.ewm(span=ewm_span).std()

    ax.plot(y, label='original')
    ax.plot(ewm_mean, label='ema')
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("10^4kW", fontsize=18)
    
    # 標準偏差から n 倍以上外れているデータを外れ値としてプロットする
    ax.fill_between(y.index,
                    ewm_mean - ewm_std * threshold,
                    ewm_mean + ewm_std * threshold,
                    alpha=0.2)
    outlier = y[(y - ewm_mean).abs() > ewm_std * threshold]
    ax.scatter(outlier.index, outlier, label='outlier')
    print('outlier: ',outlier.shape)
    #    outlier.interpolate()
    # 週ごと範囲
    outlier = outlier.reindex(range(248))

    # 日にちごと範囲
    #    outlier = outlier.reindex(range(1736))
    # 時間ごと範囲
    #    outlier = outlier.reindex(range(41664))
    print(outlier)

    #    a = outlier[()]
    outlier.to_csv('outlier_week_30.csv')

    #    ax.legend()
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1))
    return fig

    #for i in outlier:
    #    print('i: ',i)
    
#for i in range(1000):
#    if i == outlier:
#        print('1')
#    else:
#        print('0')

plot_outlier(y)
plt.show()
