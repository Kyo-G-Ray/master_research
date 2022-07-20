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


tmp = [2] * 7
resultList = []
tmpList = []

# 行ごとに取得し，値のある行は 1，ない行は 0 とする配列
for row in df.itertuples():
  # 外れ値 1 のときに配列追加
  if row[6] == 1:
    tmpList = [row[1], row[2], row[3], row[4], row[5], row[6]]
    resultList.append(tmpList)

    # 前の前の外れ値 0 のときに配列の最後から一つ前に追加
    if tmp[6] == 0:
      tmpList = [tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6]]
      resultList.insert(-1, tmpList)


  # 外れ値 0 かつ前の行の外れ値 1 のときに配列追加
  elif row[6] == 0 and tmp[6] == 1:
    tmpList = [row[1], row[2], row[3], row[4], row[5], row[6]]
    resultList.append(tmpList)

  # 一時配列保存
  tmp = row


print(resultList)