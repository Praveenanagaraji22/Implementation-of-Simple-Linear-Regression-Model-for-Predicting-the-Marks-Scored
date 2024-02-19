# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Loading: Read CSV file into DataFrame.
2. Data Splitting: Split data into training and testing sets.
3. Model Training: Train Linear Regression model.
4. Evaluation and Visualization: Calculate metrics and plot regression line.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRAVEENA N
RegisterNumber: 21222040122
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/admin/Downloads/student_scores.csv")
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
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred

Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

![1](https://github.com/anu-varshini11/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138969827/430f95db-1320-4c65-b614-ce59c4b4242e)

![2](https://github.com/anu-varshini11/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138969827/70cc133b-86fc-4bf4-8f08-8617d3eae4a7)

![3](https://github.com/anu-varshini11/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138969827/d37a1b1a-c20f-45cc-a78c-80a75865052f)

![4](https://github.com/anu-varshini11/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138969827/6357d3f1-a268-443f-8056-d6a1085818a7)

![5](https://github.com/anu-varshini11/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138969827/cc3c2c8c-6398-4e50-84bd-deebcf1b2292)

![6](https://github.com/anu-varshini11/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138969827/889f877f-4d23-4fd7-8c19-c42caefdb7c2)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
