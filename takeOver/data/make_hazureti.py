import pandas as pd
import numpy as np
from pyparsing import col




# 時間ごと
df = pd.read_csv('../output/outlier_time.csv', header=0, usecols=[1])

# # 日ごと
# df = pd.read_csv('../output/outlier_day.csv', header=0, usecols=[1])

# 週ごと
# df = pd.read_csv('../output/outlier_week_30.csv', header=0, usecols=[1])


# 外れ値を取得
hazure = df['hazureti'].values

# 出力用配列
hazureOrNot = []

# 行ごとに取得し，値のある行は 1，ない行は 0 とする配列
for row in df.itertuples():
  if not (np.isnan(row[1])):
    hazureOrNot.append(1)

  else:
    hazureOrNot.append(0)


# 外れ値かどうかを表した csv 出力．できた列を week_hazureti.csv の一番右の列に手動で追加（そのプログラム作る暇がなかった）
df['HAZURE'] = hazureOrNot


# 時間ごと
df.to_csv('hazureOrNot_time.csv', index=False)

# 日ごと
# df.to_csv('hazureOrNot_day.csv', index=False)

# 週ごと
# df.to_csv('hazureOrNot_week.csv', index=False)