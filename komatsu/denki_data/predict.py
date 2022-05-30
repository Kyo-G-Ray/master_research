import numpy
import matplotlib.pyplot as plt
import pandas
import math
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn import preprocessing

#データ読み込み　Yは最初の列に配置する
dataframe = pandas.read_csv('df.csv', usecols=[0,1,2,3,4])
dataframe['timestamp'] = dataframe['timestamp'].map(lambda _: pandas.to_datetime(_))
dataframe.set_index('timestamp', inplace=True)

print(dataframe)


model = keras.Sequential()
a = model.predict(dataframe)
print(a)
