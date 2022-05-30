import pandas as pd
import re

df = pd.read_csv('t_20202.csv',skiprows=1, header=None)
#df = pd.read_csv('t_20162.csv', encoding='Shift_JIS', skiprows=1, header=None)
#df = re.sub('時々*','',df)
print(df)

#df = df_1.rstrip('時々')
#print(df)

a = df.replace('快晴','0')
b = a.replace('晴','0')
c = b.replace('曇','0.5')
d = c.replace('薄曇','0.5')
e = d.replace('大風','0.3')
f = e.replace('霧','0.6')
g = f.replace('霧雨','0.7')
h = g.replace('雨','1.0')
i = h.replace('大雨','1.0')
j = i.replace('暴風雨','1.0')
k = j.replace('みぞれ','1.0')
l = k.replace('雪','1.0')
m = l.replace('大雪','1.0')
n = m.replace('暴風雪','1.0')
o = n.replace('地ふぶき','1.0')
p = o.replace('ふぶき','1.0')
q = p.replace('ひょう','1.0')
r = q.replace('あられ','1.0')
s = r.replace('雷','1.0')
t = s.replace('×','1.0')

print(t)
t.to_csv('2020_weather.csv')
