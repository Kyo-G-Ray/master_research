#データの読み込み
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("aweek.csv", usecols=[1,2,3,4])
pd.options.display.precision = 4
print(df.mean(axis='rows'))
df_mean = df-df.mean(axis='rows')
#print(df_mean)

a = df_mean.iloc[:,1].abs()
print(a)
a.to_csv('a.csv')


