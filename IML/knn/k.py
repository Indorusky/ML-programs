import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("data.csv")

X=data[['x','y']]
y=data['label']

model=KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)

x=(int(input("Enter the x:")))
y1=(int(input("Enter the y:")))

p=model.predict([[x,y1]])
print("Predicted:",p)

plt.scatter(X['x'],X['y'])
plt.scatter(x,y1,color='red',marker="*",s=200)
plt.show()
