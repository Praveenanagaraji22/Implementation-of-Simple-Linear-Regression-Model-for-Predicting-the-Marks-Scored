# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1. Start the program.

STEP 2.Import the standard Libraries.

STEP 3.Set variables for assigning dataset values.

STEP 4.Import linear regression from sklearn.

STEP 5.Assign the points for representing in the graph.

STEP 6.Predict the regression for marks by using the representation of the graph.

STEP 7.End the program.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRAVEENA N
RegisterNumber: 212222040122
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('/content/student_scores.csv')
print('df.head')

df.head()

print("df.tail")
df.tail()

X=df.iloc[:,:-1].values
print("Array of X")
X

Y=df.iloc[:,1].values
print("Array of Y")
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

print("Values of Y prediction")
Y_pred

print("Values of Y test")
Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Training Set Graph")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Test Set Graph")
plt.show()

print("Values of MSE, MAE and RMSE")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
## Output:
df.head()

![EXP2-2(a)](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/83e836a5-4565-48a7-9977-a7f085fe48d5)

df.tail()

![EXP2-2(b)](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/1e89f794-316e-495d-8cc6-d996bf4aa8fb)

Array value of X

![EXP2-3(a)](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/6f0f203a-f964-45a6-8fef-03c94460ab0f)

Array value of Y

![EXP2-3(b)](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/b395273b-5b41-401e-b75e-ece609b0ba79)

Values of Y Prediction

![EXP2-3(c)](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/e3e7d3bb-9a18-4345-a77c-a82898b9bee6)

Array Values of Y test

![EXP2-3(d)](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/e0268633-13d2-49c9-9a4a-b970839acf42)

Graph For Training Set

![EXP2-4](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/1c9e8b15-9199-4037-b22f-f026defe8889)

Graph For Testing Set

![EXP2-5](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/f9ed7cc5-e28a-4376-aac4-efcd9cdc92f9)

Error

![EXP2-6](https://github.com/AnnBlessy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477835/80090da9-868b-4854-b970-840396a4be77)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
