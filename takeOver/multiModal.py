import numpy
import matplotlib.pyplot as plt
import pandas
import math
import sys
import csv
from pyparsing import nums
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, concatenate, Dense

from sklearn import preprocessing
import warnings
warnings.simplefilter('ignore')



# 使用データ・層数・ニューロン数定義
whichModel = input('LSTM → l or RNN → r: ')

if whichModel == 'l':
    whichModel = 'lstm'
elif whichModel == 'r':
    whichModel = 'rnn'
else:
    sys.exit("input 'l' or 'r'")

whichData = input('データ (t or d or w): ')
numSou = input('層数 (1 or 2 or 3): ')
numSou = int(numSou)
numNeuron = input('ニューロン数 (75〜200): ')
numNeuron = int(numNeuron)




#データ読み込み Yは最初の列に配置する

# 時間ごと
if whichData == 't':
    dataframe = pandas.read_csv('./data/hazure_time.csv', usecols=[0,1,5])

# 日ごと
elif whichData == 'd':
    dataframe = pandas.read_csv('./data/hazure_day.csv', usecols=[0,1,5])

# 週ごと
elif whichData == 'w':
    dataframe = pandas.read_csv('./data/hazure_week.csv', usecols=[0,1,5])




# 時間ごと（他要因）
if whichData == 't':
    dataframe2 = pandas.read_csv('./data/hazure_time.csv', usecols=[0,2,3,4])

# 日ごと（他要因）
elif whichData == 'd':
    dataframe2 = pandas.read_csv('./data/hazure_day.csv', usecols=[0,2,3,4])

# 週ごと（他要因）
elif whichData == 'w':
    dataframe2 = pandas.read_csv('./data/hazure_week.csv', usecols=[0,2,3,4])



dataframe['timestamp'] = dataframe['timestamp'].map(lambda _: pandas.to_datetime(_))
dataframe.set_index('timestamp', inplace=True)
print(dataframe.head())

dataframe2['timestamp'] = dataframe2['timestamp'].map(lambda _: pandas.to_datetime(_))
dataframe2.set_index('timestamp', inplace=True)
print(dataframe2.head())




dataset = dataframe.values
dataset = dataset.astype('float32')

dataset2 = dataframe2.values
dataset2 = dataset2.astype('float32')


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

scaler2 = MinMaxScaler(feature_range=(0, 1))
dataset2 = scaler2.fit_transform(dataset2)


# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

train_size2 = int(len(dataset2) * 0.8)
test_size2 = len(dataset2) - train_size2
train2, test2 = dataset2[0:train_size2,:], dataset2[train_size2:len(dataset2),:]
print(len(train2), len(test2))



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


look_back = 12
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print('testX.shape',testX.shape)
print('testY.shape',testY.shape)
print('testX[0]',testX[0])
print('testY[0]',testY[0])
print('testX',testX)
print('testY',testY)

trainX2, trainY2 = create_dataset(train2, look_back)
testX2, testY2 = create_dataset(test2, look_back)
print('testX2.shape',testX2.shape)
print('testY2.shape',testY2.shape)
print('testX2[0]',testX2[0])
print('testY2[0]',testY2[0])
print('testX2',testX2)
print('testY2',testY2)


trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

trainX2 = numpy.reshape(trainX2, (trainX2.shape[0], trainX2.shape[1], trainX2.shape[2]))
testX2 = numpy.reshape(testX2, (testX2.shape[0], testX2.shape[1], testX2.shape[2]))

# create and fit the LSTM network
hidden_neurons = numNeuron
batch_size = 32



input1 = Input(shape=(3,), name='elec')
input2 = Input(shape=(4,), name='other_factor')

if whichModel == 'lstm':
    # 1 層の時
    if numSou == 1:
        x = layers.Embedding(output_dim=3, input_dim=2)(input1)
        x = layers.LSTM(hidden_neurons)(x)

        y = layers.Embedding(output_dim=3, input_dim=2)(input2)
        y = layers.LSTM(hidden_neurons)(y)


    # 2 層の時
    elif numSou == 2:
        x = layers.LSTM(hidden_neurons)(input1)
        x = layers.LSTM(hidden_neurons)(x)

        y = layers.LSTM(hidden_neurons)(input2)
        y = layers.LSTM(hidden_neurons)(y)


    # 3 層の時
    elif numSou == 3:
        x = layers.LSTM(hidden_neurons)(input1)
        x = layers.LSTM(hidden_neurons)(x)
        x = layers.LSTM(hidden_neurons)(x)

        y = layers.LSTM(hidden_neurons)(input2)
        y = layers.LSTM(hidden_neurons)(y)
        y = layers.LSTM(hidden_neurons)(y)


elif whichModel == 'rnn':
    # 1 層の時
    if numSou == 1:
        model.add(keras.layers.SimpleRNN(hidden_neurons, input_shape=(testX.shape[1], look_back)))


    # 2 層の時
    elif numSou == 2:
        model.add(keras.layers.SimpleRNN(hidden_neurons, input_shape=(testX.shape[1], look_back), return_sequences=True))
        model.add(keras.layers.SimpleRNN(hidden_neurons, input_shape=(testX.shape[1], look_back)))


    # 3 層の時
    elif numSou == 3:
        model.add(keras.layers.SimpleRNN(hidden_neurons, input_shape=(testX.shape[1], look_back), return_sequences=True))
        model.add(keras.layers.SimpleRNN(hidden_neurons, input_shape=(testX.shape[1], look_back), return_sequences=True))
        model.add(keras.layers.SimpleRNN(hidden_neurons, input_shape=(testX.shape[1], look_back)))


model = layers.concatenate([x, y])
output = layers.Dense(1, name="output")(model)

model = keras.Model(
    inputs=[x, y],
    outputs=[output],
)


model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        "output": keras.losses.BinaryCrossentropy(from_logits=True),
    },
    loss_weights=[1.0, 0.5],
)

model.fit(
    # {"elec": trainX, "other_factor": trainX2},
    # {"output": output},
    [trainX,trainX2],
    [trainY,trainY2],
    epochs=50,
    batch_size=batch_size,
)

# #optimizer を設定
# if whichModel == 'lstm':
#     opt = keras.optimizers.Adam(lr=0.001)
# elif whichModel == 'rnn':
#     opt = keras.optimizers.Adam(lr=0.00001)
# model.compile(loss='mean_squared_error', optimizer=opt)



# #callback で学習終了条件を追加
# call = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

# # time
# if whichData == 't':
#     history = model.fit(trainX, trainY, epochs=1000, batch_size=64, verbose=1,callbacks=[call], validation_split=0.1)

# # day
# if whichData == 'd':
#     history = model.fit(trainX, trainY, epochs=1000, batch_size=3, verbose=1,callbacks=[call], validation_split=0.1)

# # week
# if whichData == 'w':
#     history = model.fit(trainX, trainY, epochs=1000, batch_size=1, verbose=1, callbacks=[call], validation_split=0.1)

# #callback なし
# #history = model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=1, validation_split=0.1)



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
plt.savefig('fig/' + whichModel + '/' + whichModel + '_' + str(whichData) + '_'+ str(numSou) + '_' + str(numNeuron) + '_gosakyokusen.eps')
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
predict.to_csv('./' + whichModel + '/' + whichModel + '_' + str(whichData) + '_'+ str(numSou) + '_' + str(numNeuron) + '_predict.csv')

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.savefig('fig/' + whichModel + '/' + whichModel + '_' + str(whichData) + '_'+ str(numSou) + '_' + str(numNeuron) + '_test.eps')
plt.show()











# csv の右端に hazure 列を追加

if whichData == 't':
    dataName = 'time'
elif whichData == 'd':
    dataName = 'day'
elif whichData == 'w':
    dataName = 'week'

# csvファイルを指定
hazurePath = './data/hazureOrNot_' + dataName + '.csv'
outputPath = './' + whichModel + '/' + whichModel + '_' + str(whichData) + '_'+ str(numSou) + '_' + str(numNeuron) + '_predict.csv'


hazureList = []
count = 0
allList = []

# csv 一行ずつ読み込んで配列追加
with open(hazurePath) as f:
    reader = csv.reader(f)

    for row in reader:
        hazureList.append(row[1])


# 電力使用量と外れ値のデータ結合
with open(outputPath) as f:
    reader = csv.reader(f)

    for row in reader:
        # 変数に配列格納
        rowTmp = row
        # 配列右端に外れ値追加
        rowTmp.append(hazureList[count])
        # 全行のデータ
        allList.append(rowTmp)
        count += 1


# csv 出力
with open(outputPath, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(allList)