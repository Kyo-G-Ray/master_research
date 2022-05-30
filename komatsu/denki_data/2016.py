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
url_2016 = "https://www.tepco.co.jp/forecast/html/images/juyo-2016.csv"

df_2016 = pd.read_csv('df_2016.csv', encoding='cp932')
print(df_2016)

df_2016.to_csv('2016_2.csv',index=True)