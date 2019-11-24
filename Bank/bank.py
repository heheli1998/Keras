import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import  KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from pandas import read_csv
#导入数据并转化为数字
dataset = read_csv('E:/NLP/Keras/data/bank/bank.csv',delimiter=';')
dataset['job'] = dataset['job'].replace(to_replace=['adam.','unkonwn','unemployed','management','housemaid','enrtepreneur','student','blue-collar','self-employed'
                                                    ,'retired','tenchnician','services'],value=[0,1,2,3,4,5,6,7,8,9,10,11])
dataset['marital'] = dataset['marital'].replace(to_replace=['married','single','divorced'],value=[0,1,2])
dataset['education'] = dataset['education'].replace(to_replace=['unknown','secondary','primary','tertiary'],value=[0,2,1,3])
dataset['housing'] = dataset['housing'].replace(to_replace=['no','yes'],value=[0,1])
dataset['default'] = dataset['default'].replace(to_replace=['no','yes'],value=[0,1])
dataset['contact'] = dataset['contact'].replace(to_replace=['cellular','unknown','telephone'],value=[0,1,2])
dataset['loan'] = dataset['loan'].replace(to_replace=['no','yes'],value=[0,1])
dataset['poutcome'] = dataset['poutcome'].replace(to_replace=['unknown','other','success','failure'],value=[0,1,2,3])
dataset['month'] = dataset['month'].replace(to_replace=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],value=[1,2,3,4,5,6,7,8,9,10,11,12])
dataset['y'] = dataset['y'].replace(to_replace=['no','yes'],value=[0,1])

#分离输入与输出
array = dataset.values
x = array[:, 0:16]
Y = array[:, 16]

#设置随机种子
seed = 7
np.random.seed(seed)