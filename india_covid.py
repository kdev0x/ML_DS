import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# import KNeighborsClassifier
from numpy import loadtxt
from numpy import random
import numpy
data_load = loadtxt('C:/Users/PC/Downloads/pima-indians-diabetes.csv' , delimiter=',')
x = data_load[: , 0:8]
y = data_load[:,8]
 # # print("x:700:"x[700] , 'y:700:' ,'\n', y[700] , '\n' , 'data_load:', data_load[700] ,'\n',"x:600:" ,x[600] , 'y:600:' ,'\n', y[600] , '\n' , 'data_load:', data_load[600])


# predction while using tensorflow
modle = Sequential()
modle.add(Dense(12 , input_shape = (8,),activation = 'relu'))
modle.add(Dense(8, activation = 'relu'))
modle.add(Dense(1, activation = 'relu'))
modle.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics= ['accuracy'])
modle.fit(x,y , epochs = 150, batch_size = 10)
acc = modle.evaluate(x,y)
print(acc)

# kneighboor classfire predction
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# k = KNeighborsClassifier(n_neighbors= (12))
# mm = k.fit(x_train, y_train)
# predict1 = mm.predict(x_test)

# accuracy = accuracy_score(y_test, predict1)
# print(accuracy)



modle = Sequential()
modle.add(Dense(16 , input_shape = (8,),activation = 'relu'))
modle.add(Dense(8, activation = 'relu'))
modle.add(Dense(1, activation = 'relu'))
modle.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics= ['accuracy'])
modle.fit(x,y , epochs = 150, batch_size = 10)
acc1 = modle.evaluate(x,y)


modle = Sequential()
modle.add(Dense(24 , input_shape = (8,),activation = 'relu'))
modle.add(Dense(16, activation = 'relu'))
modle.add(Dense(8, activation = 'relu'))
modle.add(Dense(1, activation = 'sigmoid'))
modle.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics= ['accuracy'])
modle.fit(x,y , epochs = 150, batch_size = 10)
acc2 = modle.evaluate(x,y)


modle = Sequential()
modle.add(Dense(32 , input_shape = (8,),activation = 'relu'))
modle.add(Dense(24, activation = 'relu'))
modle.add(Dense(16, activation = 'relu'))
modle.add(Dense(8, activation = 'relu'))
modle.add(Dense(1, activation = 'sigmoid'))
modle.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics= ['accuracy'])
modle.fit(x,y , epochs = 150, batch_size = 10)
acc3= modle.evaluate(x,y)


print('final_result')
print(acc)
print(acc1)

print(acc2)
print(acc3)
