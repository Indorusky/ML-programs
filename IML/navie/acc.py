import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load dataset
data = pd.read_csv("heart.csv")

# Features and target
X = data[['age','sex','cholesterol','bp']]
y = data['target']

# Split dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

# Train model
model = GaussianNB().fit(X_train,y_train)

# Predict
pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test,pred)
TN,FP,FN,TP = cm.ravel()

# Metrics using formulas
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1 = 2*(precision*recall)/(precision+recall)

print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F1 Score:",f1)