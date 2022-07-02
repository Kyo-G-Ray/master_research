import pandas as pd

df = pd.read_csv('2016.csv', encoding='Shift_JIS')
print(df)

df.replace(快晴,1)
