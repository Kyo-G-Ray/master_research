import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データの読み込み
data = pd.read_csv('./data/hazure_time.csv')

# 入力データとラベルデータに分割
X = data[['RESULT', 'TEMP', 'RAIN', 'WEATHER']].values
y = data[['RESULT']].values

# 訓練データとテストデータに分割
split = int(0.8 * len(data))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# テンソルに変換
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# データローダーを定義
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# LSTMモデルを定義
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# ハイパーパラメータの設定
input_size = 4 # 入力データの次元
hidden_size = 32 # LSTMの隠れ層の次元
num_layers = 1 # LSTMの層数
learning_rate = 0.001 # 学習率
num_epochs = 10 # エポック数
batch_size = 32

# モデルの初期化
model = LSTM(input_size, hidden_size, num_layers)

# 損失関数と最適化手法の定義
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 訓練データの数を取得
train_size = len(train_dataset)

# モデルの訓練
for epoch in range(num_epochs):
    # エポックごとにモデルを初期化
    model.train()
    
    # エポックの損失を初期化
    epoch_loss = 0.0
    
    # 訓練データをバッチサイズごとに処理
    for i in range(0, train_size, batch_size):
        # バッチの先頭と末尾のインデックスを取得
        batch_start = i
        batch_end = min(i + batch_size, train_size)
        
        # バッチデータを取得
        X_batch, y_batch = train_dataset[batch_start:batch_end]
        
        # テンソルに変換する
        X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32)
        y_batch_tensor = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1)
        
        # 勾配を初期化
        optimizer.zero_grad()
        
        # 順伝播
        y_pred_batch = model(X_batch_tensor)
        
        # 損失を計算
        loss = criterion(y_pred_batch, y_batch_tensor)
        epoch_loss += loss.item() * (batch_end - batch_start)
        
        # 逆伝播
        loss.backward()
        
        # パラメータを更新
        optimizer.step()
    
    # エポックの平均損失を計算
    epoch_loss /= train_size
    
    # エポックごとに損失を出力
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.6f}")



# テストデータで予測値を計算する
y_pred_test = model.predict(X_test)

# テストデータの予測結果をnumpy配列からPandasのDataFrameに変換する
y_pred_test_df = pd.DataFrame(y_pred_test, columns=['predicted_power'])

# テストデータに対応する日時のリストを作成する
test_dates = pd.date_range(start='2019-01-01 01:00:00', end='2019-01-10 00:00:00', freq='H')

# 日時をインデックスとするPandasのDataFrameに変換する
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
