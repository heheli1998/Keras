from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.constraints import  maxnorm
from keras.callbacks import LearningRateScheduler
from math import pow,floor

#导入数据
dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target

seed = 7
np.random.seed(seed)

#构建模型函数
def create_model(init='glorot_uniform'):
    #构建模型
    model = Sequential()
    # model.add(Dropout(rate=0.2,input_shape=(4,)))
    model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer=init)) #kernel_constraint=maxnorm(3)))
    # model.add(Dropout(rate=0.2))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))#kernel_constraint=maxnorm(3)))
    # model.add(Dropout(rate=0.2))
    model.add(Dense(units=3, activation='softmax',kernel_initializer=init))
    learningRate = 0.1
    momentum = 0.9
    decay_rate = 0.0
    sgd = SGD(lr=learningRate, momentum=momentum, decay=decay_rate, nesterov=False)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
    #编译模型
def step_decay(epoch):
    init_lrate = 0.1
    drop = 0.5
    epoch_drop = 10
    lrate = init_lrate * pow(drop,floor(1+epoch) / epoch_drop)
    return  lrate

lrate = LearningRateScheduler(step_decay)
epochs = 200
model = KerasClassifier(build_fn=create_model,epochs=epochs,batch_size=5,verbose=1,callbacks=[lrate])
# kfold = KFold(n_splits=10,shuffle=True,random_state=seed) #交叉验证
# results = cross_val_score(model,x,Y,cv=kfold)
# print('Accuracy:%.2f%% (%.2f)' % (results.mean()*100, results.std()))
model.fit(x,Y)
