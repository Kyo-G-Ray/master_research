import csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np
import math





# 使用データ・層数・ニューロン数定義
whichData = input('データ (t or d or w): ')
numSou = input('層数 (1 or 2 or 3): ')
numSou = int(numSou)
numNeuron = input('ニューロン数 (75〜200): ')
numNeuron = int(numNeuron)

hazureOrNot = input('異常値(1)かそれ以外(0)か: ')




#csvファイルを指定
csvPath = './rnn_' + str(whichData) + '_'+ str(numSou) + '_' + str(numNeuron) + '_predict.csv'



trainList, testList = [], []
trainError, testError = 0, 0


with open(csvPath) as f:
  reader = csv.reader(f)
  csvfile_header = next(reader)

  trainTrue, trainPred = [], []
  testTrue, testPred = [], []
  for row in reader:
    if row[1] is not str and len(row[1]) != 0 and row[4] == hazureOrNot:
      val1 = float(row[1])
      val3 = float(row[3])

      if isinstance(val1, int) or isinstance(val1, float):
        trainPred.append(val1)
        trainTrue.append(val3)

    if row[2] is not str and len(row[2]) != 0 and row[4] == hazureOrNot:
      val2 = float(row[2])
      val3 = float(row[3])

      if isinstance(val2, int) or isinstance(val2, float):
        testPred.append(val2)
        testTrue.append(val3)


  trainList.append(trainPred)
  trainList.append(trainTrue)
  testList.append(testPred)
  testList.append(testTrue)





trainList = np.array(trainList, dtype=float)
testList = np.array(testList, dtype=float)


train_score1 = np.sqrt(mean_squared_error(trainList[0], trainList[1]))
test_score1 = np.sqrt(mean_squared_error(testList[0], testList[1]))

train_score2 = np.sqrt(mean_squared_error(preprocessing.minmax_scale(trainList[0]), preprocessing.minmax_scale(trainList[1])))
test_score2  = np.sqrt(mean_squared_error(preprocessing.minmax_scale(testList[0]), preprocessing.minmax_scale(testList[1])))


print(str(whichData) + '，' + str(numSou) + '層，' + str(numNeuron) + 'ニューロン')


varPred = np.var(preprocessing.minmax_scale(testList[0]))
varTrue = np.var(preprocessing.minmax_scale(testList[1]))
cov = np.cov(preprocessing.minmax_scale(testList[0]), preprocessing.minmax_scale(testList[1]))
print('予測データ分散: ', varPred)
print('正解データ分散: ', varTrue)
print('共分散: ', cov[0][1])



print('train_score1 : %.5f RMSE' % train_score1)
print('test_score1 : %.5f RMSE' % test_score1)
print('train_score2 : %.5f RMSE' % train_score2)
print('test_score2 : %.5f RMSE' % test_score2)