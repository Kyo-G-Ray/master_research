# モジュールインポート
import numpy as np
from matplotlib import pylab as plt
import pandas as pd
# 日本語に対応させる
#import io,sys
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 東京電力の電力使用量データ
url_2016 = "https://www.tepco.co.jp/forecast/html/images/juyo-2016.csv"
url_2017 = "https://www.tepco.co.jp/forecast/html/images/juyo-2017.csv"
url_2018 = "https://www.tepco.co.jp/forecast/html/images/juyo-2018.csv"
url_2019 = "https://www.tepco.co.jp/forecast/html/images/juyo-2019.csv"
url_2020 = "https://www.tepco.co.jp/forecast/html/images/juyo-2020.csv"

# データ読み込み
df_2016 = pd.read_csv(url_2016, encoding='Shift_JIS', skiprows=1)
df_2017 = pd.read_csv(url_2017, encoding='Shift_JIS', skiprows=1)
df_2018 = pd.read_csv(url_2018, encoding='Shift_JIS', skiprows=1)
df_2019 = pd.read_csv(url_2019, encoding='Shift_JIS', skiprows=1)
df_2020 = pd.read_csv(url_2020, encoding='Shift_JIS', skiprows=1)

#print(df_2018)

# データ結合
df = pd.concat([df_2016, df_2017, df_2018, df_2019, df_2020])
#print(df)

# 日付と時刻のデータを合わせて日時のデータとする
df['timestamp'] = df['DATE'] + " " + df['TIME']

# 日時のデータをシリアル値に変換する
df['timestamp'] = df['timestamp'].map(lambda _: pd.to_datetime(_))

df.set_index('timestamp', inplace=True)

# 不要な列を削除
df = df.drop(columns = ['DATE', 'TIME'])
        
#print(df)

# # データをプロット（全体の推移）
# plt.figure(figsize=(16, 4))
# plt.plot(df['実績(万kW)']['2018'])
# plt.xlabel("Date")
# plt.ylabel("GW")
# plt.show()

# # データをプロット（１か月の推移）
# plt.figure(figsize=(16, 4))
# plt.plot(df['実績(万kW)']['2018-04'])
# plt.xlabel("Date")
# plt.ylabel("GW")
# plt.show()

# # データをプロット（１日の推移）
# plt.figure(figsize=(16, 4))
# plt.plot(df['実績(万kW)']['2019-04-01'])
# plt.xlabel("Date")
# plt.ylabel("GW")
# plt.show()

# # データをプロット（1年前との比較）
# plt.figure(figsize=(16, 4))

# plt.plot(df['実績(万kW)']['2018-04'].values, label = "2018-04")
# plt.plot(df['実績(万kW)']['2019-04'].values, label = "2019-04")
# plt.xlabel("Date")
# plt.ylabel("GW")
# plt.legend()
# plt.show()

# 値を正規化
df['GW_normalized'] = df['実績(万kW)'] / np.mean(df['実績(万kW)'])
#print(df)

# # データをプロット（１日の推移）
# plt.figure(figsize=(16, 4))
# plt.plot(df["GW_normalized"]['2019-04-01'])
# plt.xlabel("Date")
# plt.ylabel("GW")
# plt.show()

# 学習対象のデータを定義（ここでは，2018年4月）
data = df['GW_normalized']['2018-04']
#print(data)

# 学習データの作成
X, Y = [], []

# 入力系列の長さ 
in_sequences = 10
# 上記の場合，長さ10の系列を入力とし，次の11番目の値を予測するように学習する

# 入出力のペアを作る
for i in range(len(data) - in_sequences):
    tmp_x = data[i:(i+in_sequences)]
    tmp_y = data[i+in_sequences]  
    X.append(tmp_x)
    Y.append(tmp_y)    
    #print("{}, x: {} -> y: {}".format(i, tmp_x, tmp_y))
    
# リスト形式をnumpy配列に変換
X = np.array(X)
Y = np.array(Y)

#print(X.shape, Y.shape)
#print("{} -> {}".format(X[0], Y[0]))

# 形状を変更
X = X.reshape(len(X), in_sequences, 1)
Y = Y.reshape(len(X), 1)
print("X: {}, Y: {}".format(X.shape, Y.shape))

# 学習データを訓練とテストに分割（8:2）
train_ratio = 0.8

train_len = int(len(X) * train_ratio)

x_train = X[:train_len]
x_test = X[train_len:]

y_train = Y[:train_len]
y_test = Y[train_len:]

print("Train, x: {}, y: {}".format(x_train.shape, y_train.shape))
print("Test, x: {}, y: {}".format(x_test.shape, y_test.shape))

#　RNNモデル
# モジュールのインポート
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import InputLayer
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM

# 学習状況のプロット関数の定義
def plot_history(history):    
    # 損失関数の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

# 学習用の関数の定義
def train(model, algorithm, dataset, batch = 10, epochs = 10):
    
    x_train, y_train, x_test, y_test = dataset
    
    # モデルのサマリ表示
    model.summary()
    
    #loss = 'mean_squared_error'
    loss = 'mape'

    # モデルのコンパイル
    model.compile(loss = loss, optimizer = algorithm)

    # 学習
    history = model.fit(x_train, y_train,
            batch_size=batch,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    
    # プロット
    plot_history(history)

    # テスト
    test(model, dataset)
    
    # 学習後の評価
    score_train = model.evaluate(x_train, y_train, verbose=0)    
    score_test = model.evaluate(x_test, y_test, verbose=0)    

    print('Train loss: ', score_train)
    print('Test  loss: ', score_test)

# テスト
def test(model, dataset):

    x_train, y_train, x_test, y_test = dataset

    # 予測結果
    y_train_predicted = model.predict(x_train)
    y_test_predicted = model.predict(x_test)    

    # プロット
    plt.figure(figsize=(16, 4))
    plt.plot(y_train, label = 'original(train)')
    plt.plot(y_train_predicted, label = 'predicted(train)')
    plt.xlabel("Date")
    plt.ylabel("GW")
    plt.legend()
    plt.show()

    plt.figure(figsize=(16, 4))
    plt.plot(y_test, label = 'original(test)')
    plt.plot(y_test_predicted, label = 'predicted(test)')
    plt.xlabel("Date")
    plt.ylabel("GW")
    plt.legend()
    plt.show()

# データのセット
dataset =  x_train, y_train, x_test, y_test

# 学習データセットの作成
def create_data(data, in_sequences = 10, train_ratio = 0.8):
    x, y = [], []

    # n_sequencesで指定した数を入力とし，次の値を出力とするペアを作る。
    for i in range(len(data) - in_sequences):
        tmp_x = data[i:(i+in_sequences)]
        tmp_y = data[i+in_sequences]  
        x.append(tmp_x)
        y.append(tmp_y)    
        #print("{}, x: {} -> y: {}".format(i, tmp_x, tmp_y))
    
    # リスト形式をnumpy配列に変換
    x = np.array(x)
    y = np.array(y)

    # 形状を変更
    x = x.reshape(len(x), in_sequences, 1)
    y = y.reshape(len(x), 1)
    print("X: {}, Y: {}".format(x.shape, y.shape))
    print("{} -> {}".format(x[0], y[0]))

    # 学習データを訓練とテストに分割（8:2）
    train_len = int(len(x) * train_ratio)
    x_train = x[:train_len]
    x_test = x[train_len:]
    y_train = y[:train_len]
    y_test = y[train_len:]
    
    print("Train, x: {}, y: {}".format(x_train.shape, y_train.shape))
    print("Test, x: {}, y: {}".format(x_test.shape, y_test.shape))

    return x_train, y_train, x_test, y_test

# 入力の長さ50のデータセット作成
dataset2 = create_data(data, in_sequences = 50)

# モデルの生成（LSTM）
def simple_lstm(in_sequences, in_out_dim = 1, h_neurons = 100):
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(None, in_sequences, in_out_dim)))
    model.add(LSTM(h_neurons, return_sequences=False))
    model.add(Dense(in_out_dim))
    model.add(Activation("linear"))

    return model

# モデルの読み込み
model = simple_lstm(in_sequences = 50, h_neurons = 500)
# 学習開始
train(model, Adam(), dataset2, epochs = 15, batch = 10)

# RNNモデルの構築
#hidden_neurons = 100

#model = Sequential()
#model.add(InputLayer(batch_input_shape=(None, in_sequences, 1)))
#model.add(SimpleRNN(hidden_neurons, return_sequences=False))
#model.add(Dense(1))
#model.add(Activation("linear"))

#model.summary()

#############################
#batch_input_shape:（バッチ数、学習データのステップ数、説明変数の数）
#return_sequences: Trueにすると入力の各系列ごとの出力を取得できる。ここでは，最後の出力のみ使用するので，False
#############################
