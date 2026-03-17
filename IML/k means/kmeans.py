import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load CSV
df = pd.read_csv("kmean.csv", encoding="latin1")  # keep encoding fix if needed

# Select features (choose any two)
X = df[["Marks", "StudyHours"]]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Add cluster labels
df["Cluster"] = labels

# Save output
df.to_csv("clustered_output.csv", index=False)
print("Saved clustered_output.csv")

# Plot
plt.figure(figsize=(7, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker='X',
    s=200
)
plt.xlabel("Marks (scaled)")
plt.ylabel("StudyHours (scaled)")
plt.title("K-Means Clustering (Students)")
plt.show()
