import pandas as pd

df = pd.read_csv('2016.csv', encoding='Shift_JIS', skiprows=5)
print(df)

df.replace('快晴','1')
#df.replace(2016/1/1,1)
print(df)
