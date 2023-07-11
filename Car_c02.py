# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import accuracy_score


 data = pd.read_csv("C:/Users\PC/Documents/data.csv")

print(data)


print(len(data))

# %%
print(data.shape)
print(data.head())

# %%
# do predction about weight of  a cat that is equal to 1300 and  voulem of 2300


# %%
#find line of best fit
a, b = np.polyfit(data['Weight'],  data['Volume'] , 1)

#add points to plot
plt.scatter(data['Weight'], data['Volume'])

#add line of best fit to plot
plt.plot(data['Weight'], a*data['Weight']+b)

plt.show()

# %%


# %%
plt.scatter(data['Weight'], data['Volume'] , c=data["CO2"],alpha=1)
plt.show

# %%


# %%
# x = weight , voulume y = c02
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data[['Weight','Volume']], data["CO2"])
y_pred = knn.predict([[1300 , 500]])
print(y_pred)
df = pd.DataFrame([1300 , 500])  
 
print(df)
accuracy =knn.score([[1300 , 500]], y_pred)
print("Accuracy:", accuracy)

# %%
rig = linear_model.LinearRegression()
rig.fit(data[['Weight','Volume']], data["CO2"])
y_pred = rig.predict([[1300 , 500]])
print(y_pred)
df = pd.DataFrame([1300 , 500])  
accuracy =rig.score([[1300 , 500]], y_pred)
print("Accuracy:", accuracy)  


