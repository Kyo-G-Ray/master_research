# モジュールインポート
import numpy as np
from matplotlib import pylab as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('aday_mean.csv')

# 日時のデータをシリアル値に変換する
#df['timestamp'] = df['timestamp'].map(lambda _: pd.to_datetime(_))
#df.set_index('timestamp', inplace=True)

df['DAY'] = df['DAY'].map(lambda _: pd.to_datetime(_))
df.set_index('DAY', inplace=True)

#df['DAY'] = pd.to_datetime(df["DAY"]).dt.strftime("%Y")
#df.set_index('DAY', inplace=True)

print(df)
#print(pd.to_datetime('DAY'))

# 気温 データをプロット（全体の推移）
plt.rcParams["font.size"] = 25
plt.rcParams["figure.figsize"] = (50, 5)
#plt.plot(range(91,366), df['TEMP']['2016'], color="black", label="2016")
#plt.plot(df['TEMP']['2017'].values, color="blue", label="2017")
#plt.plot(df['TEMP']['2018'].values, color="royalblue", label="2018")
#plt.plot(df['TEMP']['2019'].values, color="navy", label="2019")
#plt.plot(df['WEATHER']['2020'], color="black", label="2020")
#plt.plot(df['RESULT']['2016'], color="black", label="2016")
plt.plot(df['WEATHER']['2020'], color="black")
plt.xlabel("Date", fontsize=25)
#plt.ylabel("℃", fontsize=25)
#plt.ylabel("mm",fontsize=25)
plt.ylabel("Weather Numeric", fontsize=25)
#plt.ylabel("10^4kW", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
#fig = plt.figure()
#fig.subplots_adjust(bottom=0.2)
#ax = fig.add_subplot(111)
#, fontsize=18
#plt.rcParams.update({'font.size': 20})
#plt.tick_params(labelsize=18)
#plt.rcParams["font.size"] = 20

# 
#plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', borderaxespad=0, fontsize=18)
#plt.legend()
plt.show()

# 降水量 データをプロット（全体の推移）
#plt.figure(figsize=(100, 5))
#plt.plot(range(91,366), df['TEMP']['2016'].values, color="red", label="2016")
#plt.plot(df['TEMP']['2017'].values, color="blue", label="2017")
#plt.plot(df['TEMP']['2018'].values, color="royalblue", label="2018")
#plt.plot(df['TEMP']['2019'].values, color="navy", label="2019")
#plt.plot(df['TEMP']['2020'].values, color="black", label="2020")
#plt.xlabel("Date", fontsize=25)
#plt.ylabel("℃", fontsize=25)
#plt.rcParams.update({'font.size': 20})
#plt.tick_params(labelsize=25)
#plt.rcParams["font.size"] = 25
#plt.legend()
#plt.show()

# # 天気 データをプロット（全体の推移）
plt.rcParams["font.size"] = 25
plt.rcParams["figure.figsize"] = (50, 5)
# # plt.plot(range(91,366), df['WEATHER']['2016'].values, color="red", label="2016")
# # plt.plot(df['WEATHER']['2017'].values, color="blue", label="2017")
# # plt.plot(df['WEATHER']['2018'].values, color="royalblue", label="2018")
# # plt.plot(df['WEATHER']['2019'].values, color="navy", label="2019")
#plt.plot(preprocessing.minmax_scale(df['']['2020']), color="black", label="Weather")
plt.plot(preprocessing.minmax_scale(df['RESULT']['2020']), 'o', ms =3, color="green", label="Usage")
plt.plot(preprocessing.minmax_scale(df['TEMP']['2020']), 'o', ms =3, color="red", label="Temperature")
plt.plot(preprocessing.minmax_scale(df['RAIN']['2020']), 'o', ms = 3, color="blue", label="Rain")
plt.plot(preprocessing.minmax_scale(df['WEATHER']['2020']), 'o', ms =3, color="black", label="Weather")
#plt.plot(df['WEATHER']['2017'].values, 'o', ms = 5, c ="blue", label="2017")
# #plt.scatter(df['WEATHER']['2020'].values, df['DAY']['2020'].values, label="2020")
plt.xlabel("Date", fontsize=18)
plt.ylabel("")
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
#plt.tick_params(labelsize=18)
#plt.rcParams["font.size"] = 20
#plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', borderaxespad=0, fontsize=25)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=18)
#plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left', borderaxespad=0, fontsize=25)
#plt.legend()
plt.show()
