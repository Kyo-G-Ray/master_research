#データの読み込み
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score




# 使いたい列を usecols に
# df_score = pd.read_csv("./data/time.csv", usecols=[1,2,3,4])
# df_score = pd.read_csv("./data/day.csv", usecols=[1,2,3,4])
df_score = pd.read_csv("./data/week.csv", usecols=[1,2,3,4])

print(df_score.head())


# 標準化
print(preprocessing.minmax_scale(df_score))

# corr が 相関係数
df_corr = df_corr = df_score.corr()
print(df_corr)


# mi が相互情報量
df_score_2 = df_score.values
print(df_score.isnull().all())
#print(df_score_2)

#mi = normalized_mutual_info_score([0,0,1,1],[1,1,0,0])
mi = normalized_mutual_info_score(df_score_2[:,2],df_score_2[:,3])
print(mi)


#import matplotlib.pyplot as plt
#import japanize_matplotlib
#import seaborn as sns
#sns.heatmap(df_corr)

#plt.scatter(df_score[0],df_score[1])
