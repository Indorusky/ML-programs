import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data=pd.read_csv("kmean.csv")

X=data[['Marks','StudyHours']]

model=KMeans(n_clusters=3)
model.fit(X)

data['Cluster']=model.labels_

plt.scatter(X['Marks'],X['StudyHours'],c=data['Cluster'])
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],color='red',marker='*',s=200)
plt.title("KMeans")
plt.xlabel("Marks")
plt.ylabel("Studyhours")
plt.show()