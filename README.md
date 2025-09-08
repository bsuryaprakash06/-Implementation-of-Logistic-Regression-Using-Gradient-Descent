# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step-1

Load dataset, drop unnecessary columns, and convert categorical values into numeric codes.

Step-2

Split data into features (X) and target (Y), then initialize model parameters (θ).

Step-3

Train the model using gradient descent with the sigmoid function to optimize θ.

Step-4

Predict outcomes on training and new data, evaluate accuracy, and display results.

## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: B Surya Prakash
RegisterNumber:  212224230281
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
print("Name: B Surya Prakash")
print("Reg No: 212224230281")
dataset

dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)


dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
print("Name: B Surya Prakash")
print("Reg No: 212224230281")
dataset.dtypes


dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
print("Name: B Surya Prakash")
print("Reg No: 212224230281")
dataset


X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print("Name: B Surya Prakash")
print("Reg No: 212224230281")
Y

theta=np.random.randn(X.shape[1])
y=Y
```
```python
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))


def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)


accuracy=np.mean(y_pred.flatten()==y)
print("Name: B Surya Prakash")
print("Reg No: 212224230281")
print("Accuracy:",accuracy)
print(y_pred)
print("Name: B Surya Prakash")
print("Reg No: 212224230281")
print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
<img width="993" height="418" alt="image" src="https://github.com/user-attachments/assets/22cb416b-abb4-4602-a427-361e6a56575a" />

<img width="994" height="295" alt="image" src="https://github.com/user-attachments/assets/9b06dd61-279d-4fe5-897f-26ee9e955f97" />

<img width="1002" height="411" alt="image" src="https://github.com/user-attachments/assets/7f43ec12-5a81-4e1f-bb8a-bb350bc084be" />

<img width="995" height="232" alt="image" src="https://github.com/user-attachments/assets/d002500a-a1bc-4d65-99db-84804c881d6c" />

<img width="998" height="168" alt="image" src="https://github.com/user-attachments/assets/3cad20dd-4452-4a99-8d3c-dc631a2d3705" />

<img width="1005" height="144" alt="image" src="https://github.com/user-attachments/assets/11704be8-0b4e-43aa-86a9-7766cc23fb29" />

<img width="994" height="40" alt="image" src="https://github.com/user-attachments/assets/b02df19c-85cf-4045-a897-41be0dacde13" />

<img width="994" height="35" alt="image" src="https://github.com/user-attachments/assets/596bddf1-3056-43fe-a989-a6fdcfca3ba5" />




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

