import pandas as pd
import numpy as np

#Qs 1
data_csv=pd.read_csv('C:/Users/hp/Downloads/titanic.csv')

#Qs 2
data=pd.DataFrame(data_csv)

#Qs 3
data.dtypes

#Qs 4
data.columns
attr_list=list(data.columns)
print(attr_list)

#Qs 5
def get_attr_index(attr_list):
    age_index=data.columns.get_loc("age") if "age" in data.columns else None
    survival_index=data.columns.get_loc("survived") if "survived" in data.columns else None
    return age_index, survival_index
get_attr_index(attr_list)

#Qs 6
data.size

#Qs 7
data.info()

#Qs 8
data.describe()
 #from the  stats of the dataframe we can analyse that the 'count' gives the number of non-null values for each column
 #using the 'mean' 'std' we can get the mean and std deviation of each numerical column of the data frame
 #'min and 'max' gives us the minimum and maximum value from each column and 25%, 50%, 75% will give percentiles of the values in the dataframe
 
#Qs 9
for i in range(10):
     print(data.iloc[i]['age'])
print(data.isna().any())
print(data['age'].isna().sum())
data['age']=np.where(np.isnan(data['age']), np.nanmean(data['age']), data['age'])
data['age']=data['age'].fillna(data['age'].mean())

pass_above_60=data[data['age']>60]
print(pass_above_60)
pass_below_15=data[data['age']<=15]
print(pass_below_15)
age_15=(data['age']<=15).sum()
print(age_15)
pass_male_60=data[(data['age']>60)&(data['sex']=='male')]
print(pass_male_60)


data.pclass.hist()
import matplotlib.pyplot as plt
fig, ax=plt.subplots(1,2)
ax[1,1]=plt.plot(data.age)
ax[1,2].plot(data.age[900:])

