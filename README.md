# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages
2. Assigning hours To X and Scores to Y
3. Plot the scatter plot
4. Use mse,rmse,mae formmula to find 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Divya Sampath
RegisterNumber:  212221040042
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')
dataset.head()
dataset.tail()

X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)

import matplotlib.pyplot as plt
import numpy as np  # Make sure to import any necessary libraries

# Assuming you have your training and testing data (X_train, Y_train, X_test, Y_test) and regression model (reg) properly defined

# Plotting the training set
plt.scatter(X_train, Y_train, color="green")
plt.plot(X_train, reg.predict(X_train), color="red")
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Plotting the test set
plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, reg.predict(X_test), color="silver")  # Fixed Y_test to reg.predict(X_test)
plt.title("Test set (H vs S)")  # Fixed title
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE=',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE=',mae)
rmse=np.sqrt(mse)
print('RMSE=',rmse)

b=np.array([[10]])
Y_pred1=reg.predict(b)
print(Y_pred1)
```

## Output:
![image](https://github.com/divz2711/Implementation_of_Simple_Linear_Regression_Model_for_Predicting_the_Marks_Scored_Aim/assets/121245222/40931bf6-a2b6-459a-87b6-67dae8986103)
![image](https://github.com/divz2711/Implementation_of_Simple_Linear_Regression_Model_for_Predicting_the_Marks_Scored_Aim/assets/121245222/ad1278f2-af89-4ba5-82c8-f67c5c6fe0c6)
![image](https://github.com/divz2711/Implementation_of_Simple_Linear_Regression_Model_for_Predicting_the_Marks_Scored_Aim/assets/121245222/55445ce4-0e72-4b57-84ae-77f0315eb2bd)
![image](https://github.com/divz2711/Implementation_of_Simple_Linear_Regression_Model_for_Predicting_the_Marks_Scored_Aim/assets/121245222/de91d2e2-ebca-4d67-b7da-f0339f0abf20)
![image](https://github.com/divz2711/Implementation_of_Simple_Linear_Regression_Model_for_Predicting_the_Marks_Scored_Aim/assets/121245222/95313ca1-161d-478b-bed8-f8295db0232f)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
