import numpy as np
vector_array = np.arange(2.5,32.5).reshape(5,6) * 2

print(vector_array[ :, 2:3])
print("------------------------")
vector_array[ 4:5, :] = 1+np.arange(6)
print(vector_array[ 4:5, :])
print("------------------------")

print(vector_array)
names = np.array(['Khalid' , 'jlala' , 'aljohra' , 'mo3tz' ,'mustfa' , 'mohammed'])
result = np.array([20,15 , -100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 , 10 , 18 , 1])
print(names[ result < 15])
k = result[names == 'aljohra'] = -100
print(result)
vector_array_2 = np.arange(0,30).reshape(5,6) 
print(vector_array_2[0].sum())
vector_array_2[3:5]
print(vector_array_2[:, 3:5].sum())
