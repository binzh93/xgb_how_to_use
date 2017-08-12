# -*- coding: UTF-8 -*-
# 无法显示中文可用如下2行代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

import scipy as sp
import numpy as np
import xgboost as xgb
import math
import time
from sklearn import metrics
import matplotlib.pyplot as plt


print "---------------------------------------------"
print "--------------- Python Begin-----------------"
print "---------------------------------------------"
print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-x))
#参数适当调整
param = {}
param['objective'] = 'binary:logitraw'
param['eta'] = 0.03       #default=0.3，
param['max_depth'] = 7
param['eval_metric'] = 'auc'
param['silent'] = 1
param['min_child_weight'] = 1   #这个原来是100，太高了导致全为0，降下了可以
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['scale_pos_weight'] = 2
param['nthread'] = 4


#取出来的是numpy.darray结构
train_data = sp.genfromtxt('F:/horseTrain.csv',delimiter=',',skip_header=1)
test_data = sp.genfromtxt('F:/horseTest.csv',delimiter=',',skip_header=1)

x_list = [i for i in range(train_data.shape[1]-1)]
#训练集取各样本特征值，标签
train_x = train_data[:,x_list]
train_label = train_data[:,-1]  #这里是一行的
#测试集取各样本特征值，标签
test_x = test_data[:,x_list]
test_label = test_data[:,-1]   #这里是一行的

#print train_x.shape, type(train_x)
#print train_label.shape, type(train_label)
train_label = np.array(np.mat(np.mat(train_label).T))   #转换成列标签
#print train_label

#转换成xgboost的DMatrix格式
dtrain = xgb.DMatrix(train_x, label = train_label)
dtest = xgb.DMatrix(test_x)

#训练样本
bst = xgb.train(param, dtrain, 1000)   #迭代次数，迭代次数提高，精度会提高，但有上限
#bst.save_model('F:/xgb.model')    # 保存模型
#预测
test_pre = bst.predict(dtest)     #这里是一行的,test_pre:<type 'numpy.ndarray'>

#print test_pre
#print type(test_pre)

#将预测值转换成预测概率
prelen = len(test_pre)
res = []
for i in xrange(prelen):
	res.append(sigmoid(test_pre[i])) 
	if res[i]>0.5:
		res[i]=1
	else:
		res[i]=0
#print res
#将list转换成numpy.ndarray
res = np.array(res)

#print res
#print test_label.shape[0]
#print type(test_label)      #test_label:<type 'numpy.ndarray'>

#print正确率
judenum = test_label.shape[0]
count = 0
for i in xrange(judenum):
	if res[i] == test_label[i]:
		count += 1

correctRate = count*1.0/judenum
print correctRate


#print AUC score
print 'AUC score:',metrics.roc_auc_score(test_label, test_pre)


#绘制feature Ranking 图
#xgb.plot_tree(bst, num_trees=0)
xgb.plot_importance(bst)
plt.show()


print "---------------------------------------------"
print "--------------- Python END-----------------"
print "---------------------------------------------"
print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))