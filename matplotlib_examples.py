# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from numpy import random

# %%
x = np.array([0 , 6])
y = np.array([0 ,   250])

# %%
plt.plot(x,y)

# %%

x2 = random.randint(100 , size= [10])
y2 = random.randint(100 , size= [10])

plt.subplot(2,3,1)
plt.plot(x2,y2)
x3 = random.randint(100 , size= [10])
y3 = random.randint(100 , size= [10])
plt.subplot(2,3,2)

plt.plot(x3,y3)



x2 = random.randint(100 , size= [10])
y2 = random.randint(100 , size= [10])

plt.subplot(2,3,3)
plt.plot(x2,y2)
x3 = random.randint(100 , size= [10])
y3 = random.randint(100 , size= [10])
plt.subplot(2,3,4)



x2 = random.randint(100 , size= [10])
y2 = random.randint(100 , size= [10])

plt.subplot(2,3,4)
plt.plot(x2,y2)
x3 = random.randint(100 , size= [10])
y3 = random.randint(100 , size= [10])
plt.subplot(2,3,5)

plt.plot(x3,y3)

x2 = random.randint(100 , size= [10])
y2 = random.randint(100 , size= [10])

plt.subplot(2,3,6)
plt.plot(x2,y2) 


plt.plot(x2,y2 , marker = '*')
plt.plot(x2,y2 , ls = '-.'  , c='r' , lw = '2')
plt.xlabel("x,y")
plt.plot(axis='x' , c='r')
plt.plot(axis='y' , c='b')
plt.show()

# %%
plt.plot(x2,y2 , ls = '-.')
pl

# %%
plt.scatter(x2 , y2)
plt.show()


