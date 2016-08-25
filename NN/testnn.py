# coding=utf-8
import random

import math
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import *
# 建立神经网络fnn
from algos.nn.guiyihua import Normalization

fnn = FeedForwardNetwork()

# 设立三层，一层输入层（3个神经元，别名为inLayer），一层隐藏层，一层输出层
inLayer = LinearLayer(3, name='inLayer')
# hiddenLayer = SigmoidLayer(20, name='hiddenLayer0')
hiddenLayer = LSTMLayer(20, name='hiddenLayer0')
outLayer = LinearLayer(1, name='outLayer')

# 将三层都加入神经网络（即加入神经元）
fnn.addInputModule(inLayer)
fnn.addModule(hiddenLayer)
fnn.addOutputModule(outLayer)

# 建立三层之间的连接
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

# 将连接加入神经网络
fnn.addConnection(in_to_hidden)
fnn.addConnection(hidden_to_out)

# 让神经网络可用
fnn.sortModules()

from pybrain.supervised.trainers import BackpropTrainer

# 定义数据集的格式是三维输入，一维输出
DS = SupervisedDataSet(3,1)



# 往数据集内加样本点
# 假设x1，x2，x3是输入的三个维度向量，y是输出向量，并且它们的长度相同

n = 10000
x1 = []
x2 = []
x3 = []
y = []
for i in range(0 , n):
  x1.append(random.random())
  x2.append(random.random())
  x3.append(random.randrange(0, 5, 1))


for i in range(0 , n):
  y.append(5 * x1[i] * x1[i] - math.sin(x2[i]) * x2[i] - (x3[i] + 1) / 2)

x1 = Normalization(x1)
x2 = Normalization(x2)
x3 = Normalization(x3)

# y = Normalization(y)
print x3
print y



for i in range(0, len(y)):
  DS.addSample([x1[i], x2[i], x3[i]], [y[i]])

# 如果要获得里面的输入／输出时，可以用
X = DS['input']
Y = DS['target']

# 如果要把数据集切分成训练集和测试集，可以用下面的语句，训练集：测试集＝8:2
# 为了方便之后的调用，可以把输入和输出拎出来
dataTrain, dataTest = DS.splitWithProportion(0.8)
xTrain, yTrain = dataTrain['input'], dataTrain['target']
xTest, yTest = dataTest['input'], dataTest['target']

from pybrain.supervised.trainers import BackpropTrainer

# 训练器采用BP算法
# verbose = True即训练时会把Total error打印出来，库里默认训练集和验证集的比例为4:1，可以在括号里更改
trainer = BackpropTrainer(fnn, dataTrain, verbose=True, learningrate=0.01)

# maxEpochs即你需要的最大收敛迭代次数，这里采用的方法是训练至收敛，我一般设为1000
trainer.trainUntilConvergence(maxEpochs=1000)
