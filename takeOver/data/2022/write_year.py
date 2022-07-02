import csv
import math
from statistics import mode
import datetime
import numpy as np
import pandas as pd


# csv 読み込み
csv_file = open("./original_data.csv", "r", encoding="UTF-8", errors="", newline="" )
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
  electric_list = []
  temperature_list = []
  rain_list = []
  wea_list = []


  # 時間取得・気温・降水量・天気計算
  for j in range(startRow + (i - 1) * 24, startRow + i * 24, 1):

    # --- 電力使用量 ---
    electric_list.append(int(f[j][12]))


    # --- 時間 ---
    time = str('{0:%Y/%-m/%-d %-H:%M}'.format(datetime.datetime.strptime(f[j][0], '%Y/%m/%d %H:%M:%S')))
    time_list.append(time)


    # --- 気温 ---
    temperature_list.append(float(f[j][1]))


    # --- 降水量 ---
    rain_list.append(float(f[j][4]))


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


  # 出力用配列に追加
  csv_output.append([time_list, electric_list, temperature_list, rain_list, day_wea]) # 日付・温度・天気 を配列に追加




# データフレームを作成
to_list = []
to_list_temporary = []
column = ['timestamp', 'RESULT', 'TEMP', 'RAIN', 'WEATHER']


for i in range(dayCount):
  for j in range(24):
    to_list_temporary = [csv_output[i][0][j], csv_output[i][1][j], csv_output[i][2][j], csv_output[i][3][j], csv_output[i][4][j]]
    to_list.append(to_list_temporary)


df = pd.DataFrame(to_list, columns=column)
df.to_csv("./time_2022.csv", index=False)