import numpy as np
from matplotlib import pylab as plt
#import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from sklearn import preprocessing
#import sklearn
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error

df = pd.read_csv('df.csv')
print('df:',df)

# 日時のデータをシリアル値に変換する
df['timestamp'] = df['timestamp'].map(lambda _: pd.to_datetime(_))
df.set_index('timestamp', inplace=True)

# 値を正規化
#df['GW_normalized'] = df['RESULT'] / np.mean(df['RESULT'])
#df['TEMP_normalized'] = df['TEMP'] / np.mean(df['TEMP'])
#df['RAIN_normalized'] = df['RAIN'] / np.mean(df['RAIN'])
#print('df_normal:', df)

data = df.values
data = data.astype('float32')
#data = data.drop(columns = ['RESULT','TEMP','RAIN'])
print('data:',data)
print('data.shape:', data.shape)
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(data)

print(dataset)
