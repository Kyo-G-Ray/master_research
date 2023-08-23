import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# データの読み込み
df = pd.read_csv('./data/hazure_time.csv')

# データの前処理
# df['WEATHER'] = df['WEATHER'].map({0: 'sunny', 0.5: 'cloudy', 1: 'rainy'})  # 天気をカテゴリ変数に変換
data = df[['RESULT', 'TEMP', 'RAIN', 'WEATHER']].values  # 使用するデータを抽出
data = data.astype(np.float32)  # データをfloat32型に変換
dataLen = len(data)

# ハイパーパラメータの設定
input_size = 3  # 入力サイズ
hidden_size = 10  # 隠れ層のサイズ
output_size = 1  # 出力サイズ
sequence_length = 24  # シーケンスの長さ
num_layers = 2  # LSTMの層数
learning_rate = 0.01  # 学習率
num_epochs = 10  # エポック数
batch_size = 32

# 訓練データとテストデータに分割
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# データを0~1の範囲に正規化するスケーラーを定義する
scaler = MinMaxScaler(feature_range=(0, 1))




# データセットの作成
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length
        
    def __getitem__(self, index):
        x = self.data[index:index+self.sequence_length, :-1]
        y = self.data[index+self.sequence_length-1, -1]
        return x, y
    
    def __len__(self):
        return len(self.data) - self.sequence_length

train_dataset = CustomDataset(train_data, sequence_length)
test_dataset = CustomDataset(test_data, sequence_length)


# データローダーの作成
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)



# モデルの定義
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out



# モデルの初期化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size,num_layers=num_layers)

# 損失関数の定義
criterion = torch.nn.MSELoss()

# 最適化関数の定義
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# モデルの学習
train_loss_list = []
test_loss_list = []

for epoch in range(num_epochs):
    with tqdm(total=dataLen, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
        # Train
        model.train()  # モデルを訓練モードに変更
        train_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 順伝播
            outputs = model(inputs)
            loss = torch.sqrt(criterion(outputs, labels)) # RMSEに書き換え

            # 逆伝播と重みの更新
            loss.backward()
            optimizer.step()

            # 誤差の記録
            train_loss += loss.item()

            # 進捗バー
            pbar.update(batch_size)

        # 訓練データの誤差を記録
        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)

        # Test
        model.eval()  # モデルを評価モードに変更
        test_loss = 0
        with torch.no_grad():  # 勾配計算を行わないようにする
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = torch.sqrt(criterion(outputs, labels))

                test_loss += loss.item()

        # テストデータの誤差を記録
        test_loss /= len(test_loader)
        test_loss_list.append(test_loss)

    # エポックごとに誤差を表示
    print(f'Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}')




# テストデータを用いて予測を行う
X_test = test_data

# テンソルに変換する
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    y_pred_test = y_pred_test.detach().numpy()

# 予測結果をDataFrameに変換
y_pred_test_df = pd.DataFrame(y_pred_test, columns=["y_pred"])
y_pred_test_df.index.name = "id"

# スケーラーを用いて逆変換
y_pred_test_df["y_pred"] = scaler.inverse_transform(y_pred_test_df[["y_pred"]])

# CSVファイルに出力
y_pred_test_df.to_csv("y_pred_test.csv")



# 誤差曲線のプロット
epochs_range = range(1, num_epochs+1)
plt.plot(epochs_range, train_loss_list, label='Train Loss')
plt.plot(epochs_range, test_loss_list, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('fig/gosakyokusen.eps')
plt.show()


