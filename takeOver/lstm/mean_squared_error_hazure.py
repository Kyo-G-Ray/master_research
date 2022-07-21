import csv





# 使用データ・層数・ニューロン数定義
whichData = input('データ (t or d or w): ')
numSou = input('層数 (1 or 2 or 3): ')
numSou = int(numSou)
numNeuron = input('ニューロン数 (75〜200): ')
numNeuron = int(numNeuron)




#csvファイルを指定
csvPath = './lstm_' + str(whichData) + '_'+ str(numSou) + '_' + str(numNeuron) + '_predict.csv'

#csvファイルを読み込み
rows = []
trainError = 0
testError = 0


with open(csvPath) as f:
  reader = csv.reader(f)
  for row in reader:
    rows.append(row)
    if row[1] is not None:
      trainError += 111 # 二乗平均平方誤差の計算

    elif row[2] is not None:
      testError += 111



print(rows)