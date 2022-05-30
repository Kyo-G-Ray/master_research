import csv
import pandas as pd
import datetime



df = pd.read_csv('time.csv')


# 全ての配列入れる配列定義
df_all_list = [['DAY', 'RESULT', 'TEMP', 'RAIN', 'WEATHER']]


# 日ごとにループ回す関数
def date_range(start, stop, step = datetime.timedelta(1)):
  current = start
  while current <= stop:
    yield current
    current += step


# 日ごとに配列作って全体の配列に append
for date in date_range(datetime.date(2016, 4, 1), datetime.date(2022, 3, 31)):
  # dateを使った処理
  date = str(datetime.datetime.strftime(date, '%Y-%#-m-%#-d')).replace('-', '/') + ' '  # 文字列を含むか調べているため 4/1 に 4/10 や 4/11 などが含まれてしまうため，文字列末尾に半角スペース追加
  print(date)

  df_tmp = df[df['timestamp'].str.contains(str(date))]

  df_tmp_list = [
    date, # 日付
    df_tmp['RESULT'].sum(), # 結果の和
    round(df_tmp['TEMP'].mean(), 1), # 気温の平均
    df_tmp['RAIN'].sum(), # 降水量の和
    df_tmp['WEATHER'].mean() # 天候の平均
  ]


# 全体の配列に追加
  df_all_list.append(df_tmp_list)



with open('day.csv', 'w', newline="") as f:
  writer = csv.writer(f)
  writer.writerows(df_all_list)

