import csv
import pandas as pd
from datetime import date, timedelta



df = pd.read_csv('time.csv')

# 全ての配列入れる配列定義
df_all_list = [['DAY', 'RESULT', 'TEMP', 'RAIN', 'WEATHER']]

# 日ごとにループ回す関数
def date_range(start, stop, step = timedelta(1)):
  current = start
  while current <= stop:
    yield current
    current += step

# 日ごとに配列作って全体の配列に append
for date in date_range(date(2016, 4, 1), date(2021, 12, 31)):
  # dateを使った処理
  df_tmp = df[df['timestamp'].str.contains(str(date))]

  df_tmp_list = [
    date.strftime('%Y-%m-%d'), # 日付
    df_tmp['RESULT'].sum(), # 結果の和
    round(df_tmp['TEMP'].mean(), 1), # 気温の平均
    df_tmp['RAIN'].sum(), # 降水量の和
    df_tmp['WEATHER'].mean() # 天候の平均
  ]

# 全体の配列に追加
  df_all_list.append(df_tmp_list)




with open('day.csv', 'w') as f:
  writer = csv.writer(f)
  writer.writerows(df_all_list)

