import pandas as pd
import numpy as np

df = pd.read_csv('df.csv',usecols=[1,2,3])
df = df.values
#df = df.astype('float')
#print(len(df))

#a = [[]]
#print(sum(df[0:24]))
#print(sum(df[24:48]))
b = []
for i in range(0,len(df),24):
    a = sum(df[i:i+24])
#    print(type(a))
#    print(a)
#    b = []
    b.append(a)
    pd.DataFrame(b).to_csv('aday.csv')
#    print(b)


#    b = pd.Series(data=b, dtype='float')
#    b.to_csv('aday.csv')
#  print(a)
#     s = pd.Series(data=a, dtype='float')
#     print('s:',np.transpose(s))
#     s.to_csv('aday.csv')
# #print(a.shape)
