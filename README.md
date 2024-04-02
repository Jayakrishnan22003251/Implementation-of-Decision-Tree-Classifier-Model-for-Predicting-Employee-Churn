# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: L B L JAYAKRISHNAN
RegisterNumber:  212222230052
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evalution","number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2,0]])
```


## Output:


![Screenshot 2024-04-02 090817](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120232371/e8b77ee7-4174-4e49-9f31-c7ca23f5eef3)

![Screenshot 2024-04-02 090846](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120232371/f7a1a253-d389-40d9-96f1-493d4add0468)

![Screenshot 2024-04-02 090908](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120232371/5bdbcb49-14fb-4e03-b871-62b7747d4c2b)

![Screenshot 2024-04-02 090936](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120232371/e365b9ad-3e09-44f8-aac0-6571537f37af)

![Screenshot 2024-04-02 091014](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120232371/a4735448-6e91-40b7-9a0e-2bb6cc3c7808)

![Screenshot 2024-04-02 091050](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120232371/c692ab1c-818a-4597-b57c-413ac01f0b46)

![Screenshot 2024-04-02 091124](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120232371/68a430df-7383-4d09-97eb-6cbef1725a8c)

![image](https://github.com/Jayakrishnan22003251/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120232371/6792b045-f44a-417f-8569-7a21ca32c5b7)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
