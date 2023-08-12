# %%
import pandas as pd

data = pd.read_csv('/home/khalid/python_projects/melb_data.csv')


# %%
print(data.describe())
print('////////////////////////////////////////////')

print(data['Rooms'].sum())

# %%
print(data[data['Rooms']== 2])


# %%
data_1 = data.groupby('Suburb')

# %%
print(data_1.first())


