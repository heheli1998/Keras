from keras.layers import Activation,  Dense
from keras.models import Sequential
from keras import backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_val_score
import numpy as np

def softmax(x):
	return K.softmax(x)

seed = 7
np.random.seed(seed)

dataset = np.loadtxt('./data/pima-indians-diabetes.csv', delimiter=',')
x = dataset[:, 0:8]
Y = dataset[:, 8]

kfold = StratifiedKFold(n_splits=10,  random_state=seed, shuffle=True)
cvscores=[]

x_train,  x_validation, Y_train, Y_validation = train_test_split(x, Y, test_size=0.2,  random_state=seed)
for train,  validation in kfold.split(x, Y):
	model = Sequential()
	model.add(Dense(12,  input_dim=8))
	model.add(Activation('relu'))
	model.add(Dense(8))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',  optimizer='adam',  metrics=['accuracy'])

	# model.fit(x=x, y=Y, epochs=150, batch_size=10, validation_split=0.1)
	# model.fit(x_train,  Y_train,  validation_data = (x_validation, Y_validation),  epochs=100,  batch_size=50)
	model.fit(x[train], Y[train],epochs=100,batch_size=10,verbose=0)
	scores = model.evaluate(x[validation],  Y[validation],verbose=0)

	print('%s:%.2f%%' % (model.metrics_names[1],scores[1]*100))

	# print('\n%s: %.2f%%'%(model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1]*100)
print('%2.f%% (+/- %.2f%%)' % (np.mean(cvscores),  np.std(cvscores)))
