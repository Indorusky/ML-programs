import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("Tennis.csv")

le = LabelEncoder()
for col in data.columns:
    data[col] = le.fit_transform(data[col])

X = data[['Outlook','Temp','Humidity','Wind']]
y = data['Play']

model = DecisionTreeClassifier()
model.fit(X,y)

print("Model Trained Successfully")