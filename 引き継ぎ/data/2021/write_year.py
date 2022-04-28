import csv
import math
from statistics import mode
import datetime
import numpy as np
import pandas as pd


# csv 読み込み
csv_file = open("./weather.csv", "r", encoding="ms932", errors="", newline="" )
#リスト形式
csv = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
f = [row for row in csv]


# 開始行
startRow = 5
# 日数カウント
dayCount = math.floor((len(f) - startRow) / 24)
# csv 出力用の配列定義
csv_output = []


# 日付・気温・天気を出力
for i in range(1, dayCount + 1, 1):

  time_list = []
  temperature_list = []
  wea_list = []

  # 時間取得・気温・降水量・天気計算
  for j in range(startRow + (i - 1) * 24, startRow + i * 24 - 1, 1):
    # --- 時間 ---
    time_list.append(f[j][0])


    # --- 気温 ---
    temperature_list.append(float(f[j][1]))


    # --- 天気 ---
    weather = f[j][8]

    if(len(weather) != 0):
      wea_list.append(int(weather))


  # print(wea_list)
  most = mode(wea_list)

  if(most):
    # 晴れ
    if(most == 1 or most == 2):
      day_wea = [0] * 24

    # 曇
    elif(most == 3 or most == 4):
      day_wea = [0.5] * 24

    # 霧
    elif(most == 8):
      day_wea = [0.6] * 24

    # 霧雨
    elif(most == 9):
      day_wea = [0.7] * 24

    # 雨
    elif(most == 7 or most == 10 or most == 11 or most == 12 or most == 13 or  most == 14 or most == 15):
      day_wea = [1] * 24

    # その他の場合は曇りにしておく
    else:
      day_wea = [0.5] * 24


  # とりあえず曇りにしておく
  else:
    day_wea = [0.5] * 24


  # 超ウルトラスーパーめちゃんこ汚いが，日付を取得し，YYYY-MM-DD にした．解読しないほうがよい．
  # date = str(datetime.datetime.strptime(f[startRow + (i - 1) * 24][0].split()[0], '%Y/%m/%d')).split()[0]
  # 気温
  temperature = format(np.average(temperature_list), '.1f')
  day_temperature = [temperature] * 24

  # 出力用配列に追加
  csv_output.append([time_list, day_temperature, day_wea]) # 日付・温度・天気 を配列に追加



# for ff in range(3):
#   print(csv_output[ff], '\n')



# f = open("./data_2021.csv", mode="w", newline="")

# writer = csv.writer(f)
# for data in csv_output:
#   writer.writerow(data)
# f.close()


# データフレームを作成
to_list = []
to_list_tmp = []
column = ['DAY', 'TEMP', 'WEATHER']

for i in range(dayCount):
  for j in range(23):
    to_list_tmp = [csv_output[i][0][j], csv_output[i][1][j], csv_output[i][2][j]]
    to_list.append(to_list_tmp)
# print(to_list)
df = pd.DataFrame(to_list, columns=column)
df.to_csv("./data_2021.csv", index=False)