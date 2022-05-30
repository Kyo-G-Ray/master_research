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



#データ読み込み Yは最初の列に配置する
dataframe = pandas.read_csv('./data/week_hazureti.csv', usecols=[0,1,5])
#dataframe['timestamp'] = dataframe['timestamp'].map(lambda _: pandas.to_datetime(_))
#dataframe.set_index('timestamp', inplace=True)
dataframe['DAY'] = dataframe['DAY'].map(lambda _: pandas.to_datetime(_))
dataframe.set_index('DAY', inplace=True)
#print('df_serial:', df)
#plt.plot(dataframe)
#plt.show()
print(dataframe.head())


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
print('testX.shape',testX.shape)
print('testY.shape',testY.shape)
print('testX[0]',testX[0])
print('testY[0]',testY[0])
print('testX',testX)
print('testY',testY)


# reshape input to be [samples, time steps(number of variables), features] *convert time series into column
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

# create and fit the LSTM network
hidden_neurons = 315


model = keras.Sequential()
#model.add(keras.layers.LSTM(hidden_neurons, input_shape=(testX.shape[1], look_back)))	#shape：変数数、遡る時間数


#2 層の時
#model.add(keras.layers.LSTM(hidden_neurons, input_shape=(testX.shape[1], look_back), return_sequences=True))
#model.add(keras.layers.LSTM(hidden_neurons, input_shape=(testX.shape[1], look_back)))


#3 層の時
model.add(keras.layers.LSTM(hidden_neurons, input_shape=(testX.shape[1], look_back), return_sequences=True))
model.add(keras.layers.LSTM(hidden_neurons, input_shape=(testX.shape[1], look_back), return_sequences=True))
model.add(keras.layers.LSTM(hidden_neurons, input_shape=(testX.shape[1], look_back)))

model.add(keras.layers.Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam')
model.add(keras.layers.Activation("relu"))
#model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)


model.summary()

#optimizer を設定
opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)



#callback で学習終了条件を追加
call = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

# time
#history = model.fit(trainX, trainY, epochs=1000, batch_size=64, verbose=1,callbacks=[call], validation_split=0.1)

# day
#history = model.fit(trainX, trainY, epochs=1000, batch_size=3, verbose=1,callbacks=[call], validation_split=0.1)

# week
history = model.fit(trainX, trainY, epochs=1000, batch_size=1, verbose=1, callbacks=[call], validation_split=0.1)

#callback なし
#history = model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=1, validation_split=0.1)



# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
pad_col = numpy.zeros(dataset.shape[1]-1)


# #予測結果の保存
# tra = pandas.DataFrame(trainPredict)
# tra.to_csv('./lstm/lstm_3_315_tra.csv')
# tes = pandas.DataFrame(testPredict)
# tes.to_csv('./lstm/lstm_3_315_tes.csv')


# 誤差曲線をプロット
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.savefig('fig/lstm_gosakyokusen.eps')
plt.show()


# invert predictions
def pad_array(val):
    return numpy.array([numpy.insert(pad_col, 0, x) for x in val])


trainPredict = scaler.inverse_transform(pad_array(trainPredict))
trainY = scaler.inverse_transform(pad_array(trainY))
testPredict = scaler.inverse_transform(pad_array(testPredict))
testY = scaler.inverse_transform(pad_array(testY))


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.5f RMSE' % (trainScore))
#testY.delete(testY.columns[numpy.isnan(testY).any()], axis=1)
#testPredict.delete(testPredict.columns[numpy.isnan(testPredict).any()], axis=1)
#print('testY[:,0]',testY[:,0])
#print('testPredict[:,0]',testPredict[:,0])
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('Test Score: %.5f RMSE' % (testScore))


#入力と予測値を 0 から 1 にして誤差計算
#print('testY[:,0]:',testY[:,0])
#print('testY[:,0].shape:',testY[:,0].shape)
#print('testY[:,0]:',preprocessing.minmax_scale(testY[:,0]))


trainScore_2 = math.sqrt(mean_squared_error(preprocessing.minmax_scale(trainY[:,0]), preprocessing.minmax_scale(trainPredict[:,0])))
print('Train Score_2: %.5f RMSE' % (trainScore_2))


testScore_2 = math.sqrt(mean_squared_error(preprocessing.minmax_scale(testY[:,0]), preprocessing.minmax_scale(testPredict[:,0])))
print('Test Score_2: %.5f RMSE' % (testScore_2))



#print(testY[:,0])
#print(testPredict[:,0])

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

#予測結果の保存
original = pandas.DataFrame(scaler.inverse_transform(dataset), columns=['elec', 'what']).iloc[:,0]
tra = pandas.DataFrame(trainPredictPlot, columns=['elec_tra', 'what']).iloc[:,0]
# tra.to_csv('./lstm/lstm_3_315_tra_real.csv')
tes = pandas.DataFrame(testPredictPlot, columns=['elec_tes', 'what']).iloc[:,0]
# tes.to_csv('./lstm/lstm_3_315_tes_real.csv')
predict = pandas.concat([tra, tes, original], axis='columns')
predict.to_csv('./lstm/lstm_3_315_predict.csv')

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.savefig('fig/lstm_test.eps')
plt.show()
