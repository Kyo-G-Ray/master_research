import pandas as pd
import numpy as np
from pyparsing import col



# 時間ごと
# df = pd.read_csv('./hazure_time.csv', header=0, usecols=[0,1,2,3,4,5])

# 日ごと
# df = pd.read_csv('./hazure_day.csv', header=0, usecols=[0,1,2,3,4,5])

# 週ごと
df = pd.read_csv('./hazure_week.csv', header=0, usecols=[0,1,2,3,4,5])



# 外れ値を取得
hazure = df['HAZURE'].values

# 出力用配列
hazureOrNot = []

tmp = 0

# 行ごとに取得し，値のある行は 1，ない行は 0 とする配列
for row in df.itertuples():
  if row[6] == 1:
    # hazureOrNot.append(1)
    print(row[6])

    tmp = row[6]

    if tmp == 0:
      print('aaa')

  # else:
  #   # hazureOrNot.append(0)