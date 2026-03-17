import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("data.csv")

X = data[['x','y']]
y = data['label']

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)

x = float(input("Enter x: "))
y1 = float(input("Enter y: "))

p = model.predict([[x,y1]])

print("Prediction:",p[0])

plt.scatter(X['x'],X['y'])
plt.scatter(x,y1,color='red',marker='*',s=200)
plt.title("Prediction: "+str(p[0]))
plt.show()