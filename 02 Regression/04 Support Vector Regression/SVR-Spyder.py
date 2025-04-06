import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values
Y=Y.reshape(-1,1)



from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()

X=sc_x.fit_transform(X)
Y=sc_y.fit_transform(Y)


from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,Y)


Y_pred=regressor.predict([[6.5]])
Y_pred


plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Salary vs Position - SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



#%%
