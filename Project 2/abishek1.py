import numpy as np
#Qs 1
a=np.random.randint(10, size=(4,2))
print(a.shape)
print(a.ndim)

#Qs 2
b=np.arange(100,200,10).reshape(5,2)
print(b)

#Qs 3
c=np.array([[1.0,2.5,3.0],[2.1,3.1,4.1],[6.0,7.0,6.5]], dtype=np.float16)
print(c)

#Qs 4
d=np.array([[3 ,6, 9, 12],[15 ,18, 21, 24],[27 ,30, 33, 36],[39 ,42, 45, 48],[51 ,54, 57, 60]])
result=d[::2, 1::2]
print(result)

#Qs 5
e=np.arange(10,34,1).reshape(8,3)
sub_e=np.array_split(e, 4)
print(e)
for i in sub_e:
    print(sub_e)

#Qs 6    
one_dim = np.arange(1,6)
#array([1,2,3,4,5])
one_dim + np.arange(5, 0, -1)
#array([6,6,6,6,6])

#Qs 7
import pandas as pd
data=pd.read_excel('C:/Users/hp/Downloads/titanic.csv.xlsx')
print(data)
rows, columns=data.shape
print(data.shape[0])
print(data.shape[1])
print(data['survived'].sum())
print(data[data['sex']=='male'].shape[0])
print(data[data['sex']=='female'].shape[0])
print(data[(data['age']>40)&(data['survived']==1)].shape[0])
print(data.isnull().values.any())
data.fillna(pd.NA, inplace=True)
print(data.isnull().values.any())
