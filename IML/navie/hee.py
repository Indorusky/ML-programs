import pandas as pd
from sklearn.naive_bayes import GaussianNB

data=pd.read_csv("heart.csv")

X=data[['age','sex','cholesterol','bp']]
y=data['target']

model=GaussianNB().fit(X,y)

age=int(input("Enter the Age:"))
sex=int(input("Enter the Sex:"))
chol=int(input("Enter the Cholesterol:"))
bp=int(input("Enter the BP:"))

p=model.predict([[age,sex,chol,bp]])

if p[0]==1:
    print("Yes")
else:
    print("No")