import pandas as pd

#df.csvを読み込み
df = pd.read_csv('df.csv')
print(df)

#欠損値があれば TRUE, なければ FALSE
#要素ごとに欠損値か判定
print(df.isnull())
#行・列ごとにすべての要素が欠損値か判定
print(df.isnull().all())
