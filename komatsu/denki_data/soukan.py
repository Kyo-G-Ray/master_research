#データの読み込み
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
 
df = pd.read_csv("aweek.csv", usecols=[1,2,3,4])
print(df.shape)
#print(df)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(df)
#dataset = preprocessing.minmax_scale(df)
#print(dataset[:,0])
#print(dataset[:,0].shape)
a = np.ravel(dataset[:,0])
print('a:',a.shape)
b = np.ravel(dataset[:,1])
print('b:',b.shape)
c = np.ravel(dataset[:,2])
d = np.ravel(dataset[:,3])
#df_corr = np.corrcoef(a,b)
df = pd.DataFrame([a,b,c,d]).T
print('df:',df)
df_corr = df.corr()
print(df_corr)

