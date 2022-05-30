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
#url_2016 = "https://www.tepco.co.jp/forecast/html/images/juyo-2016.csv"
#url_2017 = "https://www.tepco.co.jp/forecast/html/images/juyo-2017.csv"
#url_2018 = "https://www.tepco.co.jp/forecast/html/images/juyo-2018.csv"
#url_2019 = "https://www.tepco.co.jp/forecast/html/images/juyo-2019.csv"
#url_2020 = "https://www.tepco.co.jp/forecast/html/images/juyo-2020.csv"

# データ読み込み
df_2016 = pd.read_csv('df_2016.csv', encoding='cp932')
df_2017 = pd.read_csv('df_2017.csv', encoding='cp932')
df_2018 = pd.read_csv('df_2018.csv', encoding='cp932')
df_2019 = pd.read_csv('df_2019.csv', encoding='cp932')
df_2020 = pd.read_csv('df_2020.csv', encoding='cp932')

# データ結合
df = pd.concat([df_2016, df_2017, df_2018, df_2019, df_2020])
print(df)

df.to_csv('df_2.csv',index=False)