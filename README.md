# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the modules pandas , numpy, matplotlib.pyplot and from sklearn.metrics import mean_absolute_error, mean_squared_error
2. Read the CSV file
3. Create a variable X to store the values of the independent variable
4. Create a variable Y to store the values of the dependent variable
5. Split the training and test data by importing train_test_split from sklearn.model_selection; choose the test size as 1/3 and the random state as 0
6. Use the LinearRegression function from sklearn.linear_model to predict the values
7. Display Y_pred
8. Display Y_test
9. Plot the graph for training data
10. Plot the graph for test data
    
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Praveena N 
RegisterNumber: 212222040122 
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SEC/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred

Y_test

plt.scatter(x_train,Y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlable("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_train,Y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlable("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE=',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE=',mae)


```


## Output:
![Training set](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393514/6aaaf0f7-5512-4a98-9402-43e7aa78b611)
![Training set](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393514/678852b4-7f0e-45f5-9bd7-4cf9cb8c231d)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
