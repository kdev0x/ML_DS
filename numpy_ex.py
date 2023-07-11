# %%
import numpy as np
from numpy import random
from math import sqrt

# %%
x = np.array([1,2])



# %%
x2 = random.randint(100 , size= [10])
y2 = random.randint(100 , size= [10])



# %%
print(x2 , y2)

# %%
x3 = np.arange(100)

# %%
print(x3)

# %%
x3.ndim
print(x3.size)
x3.ndim
print(x3.size)
print(x3.reshape(10,10),sep=",")
print(np.ones((3,4)))

# %%

print(x3 **100000)

# %%
a = np.array([[1 , 1 ],[ 1,2]])
b = np.array([[5,6] , [7,8]])
print(a*b)
print(a.dot(b))


# %%
print (sqrt(x3))


