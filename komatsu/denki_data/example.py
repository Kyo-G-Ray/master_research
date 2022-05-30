# モジュールインポート
import numpy as np
from matplotlib import pylab as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# データ読み込み
df = pd.read_csv('df.csv')
print('df:',df)

# 日時のデータをシリアル値に変換する
df['timestamp'] = df['timestamp'].map(lambda _: pd.to_datetime(_))
df.set_index('timestamp', inplace=True)
print('df_serial:', df)

# #データをプロット（全体の推移）
# plt.figure(figsize=(16, 4))
# plt.plot(df['RESULT']['2016'], color="red")
# plt.xlabel("Date")
# plt.ylabel("10^7W")
# plt.show()

# データをプロット（１日の推移）
#plt.figure(figsize=(16, 4))
#plt.plot(df['実績(万kW)']['2020-04-01'], color = 'red')
#plt.xlabel("Date")
#plt.ylabel("10^7W")
#plt.show()

# データをプロット（１日の推移）
#plt.figure(figsize=(16, 4))
#plt.plot(df['実績(万kW)']['2020-04-01'].values, color = 'red', label = "04-01")
#plt.plot(df['実績(万kW)']['2020-04-05'].values, color = 'blue', label = "04-05")
#plt.xlabel("Date")
#plt.ylabel("10^7W")
#plt.legend()
#plt.show()
df = df.values
df = df.astype('float32')
# 値を正規化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(df)
#df['GW_normalized'] = df['RESULT'] / np.mean(df['RESULT'])
#df['TEMP_normalized'] = df['TEMP'] / np.mean(df['TEMP'])
#df['RAIN_normalized'] = df['RAIN'] / np.mean(df['RAIN'])
print('df_normal:', df)

# 学習対象のデータを定義（ここでは，2018年4月）
data = df
#data = df['GW_normalized']
#data = data.drop(columns = ['RESULT','TEMP','RAIN'])
print('data:',data)
print('data.shape:', data.shape)
print('len(data):',len(data))

# 学習データの作成
X, Y = [], []

# 入力系列の長さ 
in_sequences = (10,3)
print('len(data) - in_sequences[0]:', len(data) - in_sequences[0])
# 上記の場合，長さ10の系列を入力とし，次の11番目の値を予測するように学習する

# 入出力のペアを作る
for i in range(len(data) - in_sequences[0]):
    tmp_x = data[i:(i+in_sequences[0])]
#    print('i+in_sequences[0]:',i+in_sequences[0])
    tmp_y = data[i+in_sequences[0]:i+in_sequences[0]+1]
    X.append(tmp_x)
    Y.append(tmp_y)    
    #print("{}, x: {} -> y: {}".format(i, tmp_x, tmp_y))
print('tmp_x:',tmp_x)
print('tmp_x.shape:',tmp_x.shape)
print('tmp_y:',tmp_y)
print('tmp_y.shape:',tmp_y.shape)
# リスト形式をnumpy配列に変換
X = np.array(X)
Y = np.array(Y)
print('X',X)
print('Y',Y)
print('X[0]',X[0])
print('X.shape, Y.shape',X.shape, Y.shape)
print("{} -> {}".format(X[0], Y[0]))

# 形状を変更
X = X.reshape(len(X), in_sequences[0], 3)
print('X.reshape:',X.shape)
print('len(X)',len(X))
Y = Y.reshape(len(X), 1, 3)
print("X: {}, Y: {}".format(X.shape, Y.shape))

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

print('x_train; ', x_train.shape)
print('y_train; ', y_train.shape)
print('x_test; ', x_test.shape[])
print('y_test; ', y_test.shape)
print('x_train[:,1]',x_train[:,:,1].shape)
print(y_test)
# モジュールのインポート
import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers

# RNNモデルの構築
hidden_neurons = 100

model = keras.Sequential()
model.add(keras.layers.InputLayer(batch_input_shape=(None, in_sequences[0], 3)))
model.add(keras.layers.SimpleRNN(hidden_neurons, return_sequences=False))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation("relu"))

model.summary()

# モデルのコンパイル
#model.compile(loss="mape", optimizer="adam")
model.compile(loss="mean_squared_error", optimizer="adam")

# 学習
history = model.fit(x_train, y_train, 
                    batch_size=1, epochs=3,
                    validation_data=(x_test, y_test))

# 誤差曲線をプロット
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

# 学習後の精度を評価
score_train = model.evaluate(x_train, y_train, verbose=0)    
score_test = model.evaluate(x_test, y_test, verbose=0)    
print('Train loss:', score_train)
print('Test loss:', score_test)

# テストデータに対する予測結果
y_train_predicted = model.predict(x_train[0])
y_test_predicted = model.predict(x_test[0])
print('y_train_predicted:',y_train_predicted)
print('y_test_predicted:',y_train_predicted)
# プロット
plt.figure(figsize=(16, 4))
plt.plot(y_train[0], label = 'original(train)')
plt.plot(y_train_predicted[0], label = 'predicted(train)')
plt.xlabel("Date")
plt.ylabel("GW")
plt.legend()
plt.show()

# プロット
plt.figure(figsize=(16, 4))
plt.plot(y_test[0], label = 'original(test)')
plt.plot(y_test_predicted[0], label = 'predicted(test)')
plt.xlabel("Date")
plt.ylabel("GW")
plt.legend()
plt.show()
