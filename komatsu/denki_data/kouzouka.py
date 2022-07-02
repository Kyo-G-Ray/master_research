from random import random

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model
import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
#import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers

#データ読み込み　Yは最初の列に配置する
dataframe = pandas.read_csv('df.csv', usecols=[0,1,2,3,4])
dataframe['timestamp'] = dataframe['timestamp'].map(lambda _: pandas.to_datetime(_))
dataframe.set_index('timestamp', inplace=True)
#print('df_serial:', df)
#plt.plot(dataframe)
#plt.show()
print(dataframe.head())
print(dataframe.shape)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
print('train: ',train)
print('test: ',test)

# convert an array of values into a dataset matrix
# if you give look_back 3, a part of the array will be like this: Jan, Feb, Mar
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        xset = []
        for j in range(dataset.shape[1]):
            a = dataset[i:(i+look_back), j]
            xset.append(a)
        dataY.append(dataset[i + look_back, 0])      
        dataX.append(xset)
    return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t and Y=t+1
look_back = 12
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
#print('trainY[:,0]',trainY[:,0])
print('trainX',trainX)
print('trainY',trainY)
print('trainX.shape',trainX.shape)
print('trainY.shape',trainY.shape)
print('testX.shape',testX.shape)
print('testY.shape',testY.shape)
print('testX[0]',testX[0])
print('testY[0]',testY[0])
print('testX',testX)
print('testY',testY)

# reshape input to be [samples, time steps(number of variables), features] *convert time series into column
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))
print('trainX',trainX[:,0].shape)
print('testX',testX.shape[1])

# 入力を定義
input1 = Input(shape=(look_back))
input2 = Input(shape=(look_back))

# 入力1から結合前まで
x = Dense(1, activation="linear")(input1)
x = Model(inputs=input1, outputs=x)

# 入力2から結合前まで
y = Dense(1, activation="linear")(input2)
y = Model(inputs=input2, outputs=y)

# 結合
combined = concatenate([x.output, y.output])

# 密結合
z = Dense(32, activation="tanh")(combined)
z = Dense(1, activation="sigmoid")(z)

# モデル定義とコンパイル
model = Model(inputs=[x.input, y.input], outputs=z)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
history = model.fit([trainX[:,0], trainX[:,1]], trainY, epochs=10)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
pad_col = numpy.zeros(dataset.shape[1]-1)
print('trainPredict :',trainPredict)
print('trainPredict.shape :',trainPredict.shape)
print('testPredict :',testPredict)
print('testPredict.shape :',testPredict.shape)


