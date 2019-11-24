from keras.layers import Activation,  Dense
from keras.models import Sequential
from keras import backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_val_score
import numpy as np
from keras.wrappers.scikit_learn import  KerasClassifier

def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',  optimizer='adam',  metrics=['accuracy'])

    return model
seed = 7
np.random.seed(seed)
#导入数据
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
#分割输入变量x和输出变量Y
x = dataset[:, 0:8]
Y = dataset[:, 8]

#创建 for scikit-learn
model = KerasClassifier(build_fn=create_model,epochs=150,batch_size=10,verbose=0)

#10折交叉验证
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results = cross_val_score(model,x,Y,cv=kfold)
print(results.mean())