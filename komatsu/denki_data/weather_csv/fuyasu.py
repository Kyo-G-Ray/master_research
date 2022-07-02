import pandas as pd
import numpy as np

df = pd.read_csv('2020_24.csv')
df = df.values
print(df.shape)

df = df.reshape([8784,1])
print(df.shape)
df = pd.DataFrame(data=df, dtype='float')
df.to_csv('2020_weather_24.csv')
#df_reshape = np.reshape(1,0)
