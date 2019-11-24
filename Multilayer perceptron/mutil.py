from sklearn import datasets
from keras.utils import to_categorical
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time

seed = 7
np.random.seed(seed)

dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target
# x_train,x_increment,Y_train,Y_increment = train_test_split(x,Y,test_size=0.2,random_state=seed)

#将标签转换成分类编码
Y_labels = to_categorical(Y,num_classes =3) #Y_train

#构建模型函数
def create_mmodel(optimizer='rmsprop',init='glorot_uniform'):
    #构建模型
    model = Sequential()
    model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer = init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

    #编译模型
    model.compile(loss='categorical_crossentropy',optimizer = optimizer,metrics= ['accuracy'])
    return model

#构建模型
model = create_mmodel()
# model.fit(x_train,Y_train_labels,epochs=10,batch_size=5,verbose=2)
history = model.fit(x,Y_labels,validation_split=0.2,epochs=200,batch_size=5,verbose=0)

scores = model.evaluate(x,Y_labels,verbose=0)
print("%s: %.2ff%%" % (model.metrics_names[1],scores[1]*100))

#History列表
print(history.history.keys())

#accuracy的历史
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc= 'upper left')
plt.show()
#loss的历史
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc= 'upper left')
plt.show()

#将模型保存成JSON文件

# model_json = model.to_json()
# with open('model.increment.json','w') as file:
#     file.write(model_json)
#
# #保存模型的权重值
# model.save_weights('model.increment.json.h5')
#
# #从json(或者改为json)文件中加载模型
# with open('model.increment.json','r') as file:
#     model_json = file.read()
#
# #加载模型
#
# new_model = model_from_json(model_json)
# new_model.load_weights('model.increment.json.h5')

#编译模型

# new_model.compile(loss='categorical_crossentropy',optimizer = 'rmsprop',metrics=['accuracy'])


# #增量训练模型
# Y_increment_labels = to_categorical(Y_increment,num_classes=3)
# new_model.fit(x_increment,Y_increment_labels,epochs=10,batch_size=5,verbose=2)
# scores= new_model.evaluate(x_increment,Y_increment_labels,verbose=0)
# print('Incement %s: %.2d%%' %(model.metrics_names[1],scores[1]*100))

#评估从json中加载的模型
#
# scores = new_model.evaluate(x,Y_labels,verbose=0)
# print('%s: %.2ff%%' % (model.metrics_names[1],scores[1]*100))

