import pandas as pd
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("heart.csv")

X = data[['age','sex','cholesterol','bp']]
y = data['target']

model = GaussianNB().fit(X,y)

age=int(input("Age: "))
sex=int(input("Sex(1=M,0=F): "))
chol=int(input("Cholesterol: "))
bp=int(input("BP: "))

p = model.predict([[age,sex,chol,bp]])

if p[0]==1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")