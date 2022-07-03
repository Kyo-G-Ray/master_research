# 参考  https://qiita.com/hoto17296/items/d337fe0215907432d754

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# df = pd.read_csv("./data/time.csv", usecols=[0,1])
df = pd.read_csv("./data/day.csv", usecols=[0,1])
# df = pd.read_csv("./data/week.csv", usecols=[0,1])
print(df.head())

# time
# x = df['timestamp'].values

# day
x = df['timestamp'].values
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
plt.xlabel("Date", fontsize=16)
plt.ylabel("10^4kW", fontsize=16)
#ax.legend()
#plt.show()


# threshold で何倍以上を外れ値として扱うかを示す
def plot_outlier(y, ewm_span=16, threshold=1.0):
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["font.size"] = 12
    y = pd.Series(y)
    fig, ax = plt.subplots()


    # 指数加重移動平均
    ewm_mean = y.ewm(span=ewm_span).mean()

    # 指数加重移動標準偏差
    ewm_std = y.ewm(span=ewm_span).std()

    ax.plot(y, label='original')
    ax.plot(ewm_mean, label='ema')
    plt.xlabel("Date", fontsize=16)
    plt.ylabel("10^4kW", fontsize=16)


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
    # outlier = outlier.reindex(range(313))
    # outlier.to_csv('output/outlier_week_30.csv')

    # 日にちごと範囲
    outlier = outlier.reindex(range(2192))
    outlier.to_csv('output/outlier_day.csv')

    # 時間ごと範囲
    # outlier = outlier.reindex(range(52585))
    # outlier.to_csv('output/outlier_time.csv')

    print(outlier)



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
plt.savefig('./fig/outlier.eps')
plt.show()