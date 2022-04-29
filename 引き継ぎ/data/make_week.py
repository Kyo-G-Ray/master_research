import pandas as pd


# csv 読み込み
df = pd.read_csv('time.csv', index_col=0, parse_dates=True)



# 7 日ずつ集計
df = df.resample('7D').sum()



# csv 出力
df.to_csv('week.csv')

