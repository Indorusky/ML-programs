import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data=pd.read_csv("heart.csv")

X=data[['age','sex','cholesterol','bp']]
y=data['target']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

model=GaussianNB().fit(X_train)

pred=model.predict(X_test)

cm=confusion_matrix(y_test,pred)
TN,TP,FP,FN=cm.revel()

precision=TP/(TP+FP)
recall=TP/(TP+FN)
accuracy=(TP+TN)/(TP+TN+FP+FN)
f1=(2*precision*recall)/(2*precision+recall)

print("Precision:",precision)
print("Accuracy:",accuracy)
print("RRecall:",recall)
print("F1:",f1)