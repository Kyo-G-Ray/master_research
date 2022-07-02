# モジュールインポート
import numpy as np
from matplotlib import pylab as plt
import pandas as pd

# モジュールのインポート
import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import layers

# 東京電力の電力使用量データ
url_2020 = "https://www.tepco.co.jp/forecast/html/images/juyo-2020.csv"
print(url_2020)

# データ読み込み
df = pd.read_csv(url_2020, encoding='Shift_JIS', skiprows=1)
print(df)

# 日付と時刻のデータを合わせて日時のデータとする
#df['timestamp'] = df['DATE'] + " " + df['TIME']
#print(df)

# 日時のデータをシリアル値に変換する
#df['timestamp'] = df['timestamp'].map(lambda _: pd.to_datetime(_))
#df.set_index('timestamp', inplace=True)
#print(df)

# 不要な列を削除
#df = df.drop(columns = ['DATE', 'TIME'])
#print(df)

# 値を正規化
#df['GW_normalized'] = df['実績(万kW)'] / np.mean(df['実績(万kW)'])
#print(df)


df.to_csv('df_2020.csv',index=True)