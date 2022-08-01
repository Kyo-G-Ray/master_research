# モジュールインポート
import numpy as np
from matplotlib import pylab as plt
import pandas as pd
import math
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
#import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn import preprocessing



# 使用データ・層数・ニューロン数定義
whichData = input('データ (t or d or w): ')
numSou = input('層数 (1 or 2 or 3): ')
numSou = int(numSou)
numNeuron = input('ニューロン数 (75〜200): ')
numNeuron = int(numNeuron)



# データ読み込み

# 時間ごと
if whichData == 't':
# 電力使用量のみ
    dataframe = pd.read_csv('./data/hazure_time.csv', usecols=[1])
# 4 項目を入力
    # dataframe = pd.read_csv('./data/hazure_time.csv',usecols=[1,2,3,4,5])

# 日ごと
elif whichData == 'd':
    dataframe = pd.read_csv('./data/hazure_day.csv', usecols=[1])

# 週ごと
elif whichData == 'w':
    dataframe = pd.read_csv('./data/hazure_week.csv', usecols=[1])

print(dataframe)


# データ結合
#df = pd.concat([df_2016, df_2017, df_2018, df_2019, df_2020])

# 日付と時刻のデータを合わせて日時のデータとする
#df['timestamp'] = df['DATE'] + " " + df['TIME']

# 時間ごとのデータをシリアル値に変換する
#dataframe['timestamp'] = dataframe['timestamp'].map(lambda _: pd.to_datetime(_))
#dataframe.set_index('timestamp', inplace=True)

# 日ごとのデータ処理
#dataframe['DAY'] = dataframe['DAY'].map(lambda _: pd.to_datetime(_))
#dataframe.set_index('DAY', inplace=True)

dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 不要な列を削除
#df = df.drop(columns = ['DATE', 'TIME'])        
#print(df)

# 値を正規化
#df['GW_normalized'] = df['RESULT'] / np.mean(df['RESULT'])

# 学習対象のデータを定義（ここでは，2018年4月）
#data = df
#data = df['GW_normalized']
#data = data.drop(columns = ['RESULT'])
print('dataset; ', dataset)



# 学習データの作成
X, Y = [], []


# 入力系列の長さ 
in_sequences = 10
# 上記の場合，長さ10の系列を入力とし，次の11番目の値を予測するように学習する



# 入出力のペアを作る
for i in range(len(dataset) - in_sequences):
    tmp_x = dataset[i:(i+in_sequences)]
    tmp_y = dataset[i+in_sequences]  
    X.append(tmp_x)
    Y.append(tmp_y)    
    #print("{}, x: {} -> y: {}".format(i, tmp_x, tmp_y))


# リスト形式をnumpy配列に変換
X = np.array(X)
Y = np.array(Y)

print(X.shape, Y.shape)
print("{} -> {}".format(X[0], Y[0]))


# 形状を変更
#X = X.reshape(len(X), in_sequences, 4)
#Y = Y.reshape(len(X), 4)
#print("X: {}, Y: {}".format(X.shape, Y.shape))


# 学習データを訓練とテストに分割（8:2）
train_ratio = 0.8

train_len = int(len(X) * train_ratio)

x_train = X[:train_len]
x_test = X[train_len:]

y_train = Y[:train_len]
y_test = Y[train_len:]

#print("Train, x: {}, y: {}".format(x_train.shape, y_train.shape))
#print("Test, x: {}, y: {}".format(x_test.shape, y_test.shape))


# 学習データを訓練とテストに分割（8:2）
#from sklearn.model_selection import train_test_split

#x_train, x_test = train_test_split(X, train_size=0.8)
#y_train, y_test = train_test_split(Y, train_size=0.8)

#print('x_train; ', x_train.shape)
#print('y_train; ', y_train.shape)
#print('x_test; ', x_test.shape)
#print('y_test; ', y_test.shape)



# モジュールのインポート
import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers



# RNNモデルの構築
hidden_neurons = numNeuron

model = keras.Sequential()

# 電力使用量のみを入力
model.add(keras.layers.InputLayer(batch_input_shape=(None, in_sequences, 1)))

# 4 項目を入力 （多分間違っとる）
# model.add(keras.layers.InputLayer(batch_input_shape=(None, in_sequences, 5)))


# 1 層の時
if numSou == 1:
    model.add(keras.layers.SimpleRNN(hidden_neurons))


# 2 層の時
elif numSou == 2:
    model.add(keras.layers.SimpleRNN(hidden_neurons, return_sequences=True))
    model.add(keras.layers.SimpleRNN(hidden_neurons))


# 3 層の時
elif numSou == 3:
    model.add(keras.layers.SimpleRNN(hidden_neurons, return_sequences=True))
    model.add(keras.layers.SimpleRNN(hidden_neurons, return_sequences=True))
    model.add(keras.layers.SimpleRNN(hidden_neurons))
    #model.add(keras.layers.SimpleRNN(hidden_neurons, return_sequences=False))



model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation("relu"))

model.summary()


# モデルのコンパイル
opt = keras.optimizers.Adam(lr=0.00001)
model.compile(loss="mean_squared_error", optimizer=opt)

#callback で学習終了条件を追加
call = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
# call = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')



# 学習

# time
if whichData == 't':
    history = model.fit(x_train, y_train,batch_size=64, epochs=1000,validation_data=(x_test, y_test),callbacks=[call])


# day
if whichData == 'd':
    history = model.fit(x_train, y_train,batch_size=3,epochs=1000,validation_data=(x_test, y_test),callbacks=[call])


# week
if whichData == 'w':
    history = model.fit(x_train, y_train,batch_size=1,epochs=1000,validation_data=(x_test, y_test),callbacks=[call])



# 誤差曲線をプロット
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.savefig('./fig/rnn_' + str(whichData) + '_'+ str(numSou) + '_' + str(numNeuron) + 'gosakyokusen.eps')
plt.show()


# 学習後の精度を評価
score_train = model.evaluate(x_train, y_train, verbose=0)    
score_test = model.evaluate(x_test, y_test, verbose=0)    
print('Train loss:', score_train)
print('Test loss:', score_test)


# テストデータに対する予測結果
y_train_predicted = model.predict(x_train)
y_test_predicted = model.predict(x_test)

tra = pd.DataFrame(y_train_predicted)
tra.to_csv('./rnn/rnn_' + str(whichData) + '_'+ str(numSou) + '_' + str(numNeuron) + '_tra.csv')
tes = pd.DataFrame(y_test_predicted)
tes.to_csv('./rnn/rnn_' + str(whichData) + '_'+ str(numSou) + '_' + str(numNeuron) + '_tes.csv')


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_predicted[:,0]))
print('Train Score: %.5f RMSE' % (trainScore))
#testY.delete(testY.columns[numpy.isnan(testY).any()], axis=1)
#testPredict.delete(testPredict.columns[numpy.isnan(testPredict).any()], axis=1)
#print('testY[:,0]',testY[:,0])
#print('testPredict[:,0]',testPredict[:,0])
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_predicted[:,0]))
print('Test Score: %.5f RMSE' % (testScore))


#入力と予測値を 0 から 1 にして誤差計算
#print('testY[:,0]:',testY[:,0])
#print('testY[:,0].shape:',testY[:,0].shape)
#print('testY[:,0]:',preprocessing.minmax_scale(testY[:,0]))


trainScore_2 = math.sqrt(mean_squared_error(preprocessing.minmax_scale(y_train[:,0]), preprocessing.minmax_scale(y_train_predicted[:,0])))
print('Train Score_2: %.5f RMSE' % (trainScore_2))

testScore_2 = math.sqrt(mean_squared_error(preprocessing.minmax_scale(y_test[:,0]), preprocessing.minmax_scale(y_test_predicted[:,0])))
print('Test Score_2: %.5f RMSE' % (testScore_2))



# プロット
plt.figure(figsize=(12, 6))
plt.plot(y_train, label = 'original(train)')
plt.plot(y_train_predicted, label = 'predicted(train)')
plt.xlabel("Date")
plt.ylabel("GW")
plt.legend()
plt.savefig('./fig/rnn/rnn_' + str(whichData) + '_'+ str(numSou) + '_' + str(numNeuron) + '_train.eps')
plt.show()


# プロット
plt.figure(figsize=(12, 6))
plt.plot(y_test, label = 'original(test)')
plt.plot(y_test_predicted, label = 'predicted(test)')
plt.xlabel("Date")
plt.ylabel("GW")
plt.legend()
plt.savefig('./fig/rnn/rnn_' + str(whichData) + '_'+ str(numSou) + '_' + str(numNeuron) + '_test.eps')
plt.show()
