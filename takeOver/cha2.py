import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # 一時的にwarningを無視するコード



# ハイパーパラメータの設定
input_size = 4 # 入力データの次元
hidden_size = 32 # LSTMの隠れ層の次元
num_layers = 1 # LSTMの層数
learning_rate = 0.001 # 学習率
num_epochs = 3 # エポック数
batch_size = 32



# データの読み込み
data = pd.read_csv('./data/hazure_time.csv')
dataCopy = data.copy()

# 入力データとラベルデータに分割
X = data[['RESULT', 'TEMP', 'RAIN', 'WEATHER']].values
y = data[['RESULT']].values

dataLen = len(data)

# データを0~1の範囲に正規化するスケーラーを定義する
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
# y = scaler.fit_transform(y)

# 訓練データとテストデータに分割
split = int(0.8 * len(data))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# 正解データを0~1の範囲に正規化するスケーラーを定義する
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train = scaler_y.fit_transform(y_train)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32).squeeze()

# テンソルに変換
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# データローダーを定義
train_dataset = MyDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MyDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# LSTMモデルを定義
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size).requires_grad_()
        if len(x.shape) == 2:
            out, (hn, cn) = self.lstm(x.unsqueeze(0), (h0, c0))
        else:
            out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# モデルの初期化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(input_size, hidden_size, num_layers)

# 損失関数と最適化手法の定義
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 訓練データの数を取得
train_size = len(train_loader.dataset)

# モデルの学習
train_loss_list = []
test_loss_list = []

# モデルの訓練
for epoch in range(num_epochs):
    bar_format = '{desc}: {percentage:3.0f}%|{bar:30}{r_bar}' + '\033[32m' # バーを緑色。ならんかったけど
    with tqdm(total=dataLen, desc=f'Epoch {epoch+1}/{num_epochs}', bar_format=bar_format, ncols=100) as pbar:
        # エポックごとにモデルを初期化
        model.train()
        
        # エポックの損失を初期化
        train_loss = 0
        
        # バッチサイズごとにデータを取得
        for X_batch, y_batch in train_loader:
            
            # 勾配を初期化
            optimizer.zero_grad()
            
            # 順伝播
            y_pred_batch = model(X_batch)
            
            # 損失を計算
            loss = criterion(y_pred_batch, y_batch.unsqueeze(1))
            train_loss += loss.item() * X_batch.shape[0]
            
            # 逆伝播
            loss.backward()
            
            # パラメータを更新
            optimizer.step()

            # # 誤差の記録
            # train_loss += loss.item()

            # 進捗バー
            pbar.update(batch_size)
        
        # # エポックの平均損失を計算
        # train_loss /= train_size

        # # 訓練データの誤差を記録
        # train_loss /= len(train_loader)
        train_loss_list.append(train_loss / train_size)

        # Test
        model.eval()  # モデルを評価モードに変更
        test_loss = 0
        y_pred_test = []
        with torch.no_grad():  # 勾配計算を行わないようにする
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = torch.sqrt(criterion(outputs, labels))

                test_loss += loss.item()

                # テストデータをモデルに入力し、予測値を取得
                y_pred_test_batch = model(X_batch.to(device)).cpu().numpy()
                # 予測値をリストに追加
                y_pred_test.append(y_pred_test_batch)

        # テストデータの誤差を記録
        test_loss /= len(test_loader)
        test_loss_list.append(test_loss)

    # エポックごとに誤差を表示
    print(f'Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}')
        
    # # エポックごとに損失を出力
    # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.6f}")




# 予測値のリストを1つの配列に変換
y_pred_test = np.concatenate(y_pred_test, axis=0)

# 予測値をスケーラーで逆正規化
y_pred_test = scaler.inverse_transform(y_pred_test)

# # テストデータの予測結果をnumpy配列からPandasのDataFrameに変換する
# y_pred_test_df = pd.DataFrame(y_pred_test, columns=['predicted_power'])

# test_datesに対応するindexを取得
test_dates = dataCopy.iloc[split:]['timestamp'].values
test_idx = dataCopy.iloc[split:].index

# y_pred_test_dfを作成
y_pred_test_df = pd.DataFrame(y_pred_test, index=test_idx, columns=['RESULT'])

# インデックスをtest_datesに設定
print(test_dates)
y_pred_test_df.index = test_dates

# 予測結果のDataFrameをcsvファイルとして保存する
y_pred_test_df.to_csv('predicted_power.csv')

# グラフを作成する
fig, ax = plt.subplots(figsize=(10, 5))

# 実測値と予測値のグラフを描画する
ax.plot(test_dates, y_test, label='actual power')
ax.plot(test_dates, y_pred_test, label='predicted power')

# グラフにタイトルや軸ラベルを設定する
ax.set_title('Predicted and actual power')
ax.set_xlabel('Date and Time')
ax.set_ylabel('Power consumption (MW)')
ax.legend()

# グラフを表示する
plt.show()
