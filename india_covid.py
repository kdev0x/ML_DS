
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from numpy import loadtxt
from numpy import random


data_load = loadtxt('C:/Users/PC/Downloads/pima-indians-diabetes.csv' , delimiter=',')

# print(data_load)

# print(data_load.shape)

# print(data_load.size)

# print(data_load[700] , '\n' , data_load.shape)
x = data_load[:,0:8]
y = data_load[:,8]
# print(x,y)
# # print("x:700:"x[700] , 'y:700:' ,'\n', y[700] , '\n' , 'data_load:', data_load[700] ,'\n',"x:600:" ,x[600] , 'y:600:' ,'\n', y[600] , '\n' , 'data_load:', data_load[600])


modle = Sequential()
modle.add(Dense(12 , input_shape = (8,),activation = 'relu'))
modle.add(Dense(8, activation = 'relu'))
modle.add(Dense(1, activation = 'sigmoid'))
modle.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
modle.fit(x,y , epochs = 150, batch_size = 10)
acc = modle.evaluate(x,y)
print(acc)
