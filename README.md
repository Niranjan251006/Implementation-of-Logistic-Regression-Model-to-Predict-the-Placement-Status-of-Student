# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NIRANJAN S
RegisterNumber:  24900209
*/
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
## PLACEMENT DATA:
![267753589-cba641d7-4b64-474a-9df3-f8047b4ddc21](https://github.com/user-attachments/assets/5d9d747f-3a63-43ed-beb2-13fecac183c3)
## SALARY DATA:
![267753686-b69592e3-fb46-446d-87a4-60e8dabf45a1](https://github.com/user-attachments/assets/c9b58d02-674e-44b8-b72b-7d610bddada1)

## CHECKING NULL() FUNCTION:
![267753782-196a08f0-0571-40f2-bfdf-b6e1d2b4fa8f](https://github.com/user-attachments/assets/700b4ff1-8fe4-4eb5-9e45-4d500dde902b)
## DUPLICATE DATA
![267753891-3efb2a8c-6c60-4466-99b2-2c3c7b7a39b4](https://github.com/user-attachments/assets/7ed55b95-6e59-4ac2-9cb8-ee97b1c18486)
## PRINT DATA
![267753963-37d05f23-2187-49d2-a871-7dbf5d7baca9](https://github.com/user-attachments/assets/b58bd481-af09-428f-b8d4-9bf61a88ffab)

## DATA STATUS:
![267754049-d0b24ebb-4d7a-4956-b6e5-b87f65ccbeeb](https://github.com/user-attachments/assets/059d17f5-45ad-4eba-91b9-73d2a9c35fd0)

## Y_PREDICTION ARRAY
![267754328-81a5cd80-1fa0-48d8-a838-567b6e7a6676](https://github.com/user-attachments/assets/21daa568-d76b-4f3e-a42e-9e300d45e21a)

## ACCURACY VALUE
![267754448-1ca21819-8baa-4312-aae8-1b094fe75ea6](https://github.com/user-attachments/assets/708df026-d450-466a-9111-5c8fe43c5c8a)

## CONFUSION ARRAY
![267754513-675efabe-006d-463a-b5f0-0cc4354ca37a](https://github.com/user-attachments/assets/d799992e-8002-4dc2-b279-f0e17f50961b)

## CLASSIFICATION REPORT
![267754597-be3ab929-d71c-492a-8adc-9a054cf08983](https://github.com/user-attachments/assets/59ee93c6-5456-40db-83a0-af814e9fb9d7)


## PREDICTION LR
![267754663-295b82c5-385c-4832-9d92-282a651946cb](https://github.com/user-attachments/assets/6abb2f4d-3e8b-4b35-a2bf-e3a9e0e07705)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
