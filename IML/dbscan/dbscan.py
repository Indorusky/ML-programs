import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# -------------------------
# 1. Load CSV
# -------------------------
df = pd.read_csv("db.csv", encoding="latin1")

# Drop unwanted unnamed columns (Excel leftovers)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# -------------------------
# 2. Select features for DBSCAN
# Use column positions to avoid header mismatch:
# (Assuming Annual Income is 4th column, Spending Score is 5th)
# -------------------------
X = df.iloc[:, [3, 4]]

# -------------------------
# 3. Scale
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 4. Apply DBSCAN
# -------------------------
dbscan = DBSCAN(eps=0.8, min_samples=3)
labels = dbscan.fit_predict(X_scaled)

# -------------------------
# 5. Save output
# -------------------------
df["Cluster"] = labels
df.to_csv("dbscan_output.csv", index=False)
print("Saved dbscan_output.csv")

# -------------------------
# 6. Plot
# -------------------------
plt.figure(figsize=(7, 5))
unique_labels = set(labels)

for label in unique_labels:
    mask = labels == label
    if label == -1:
        plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], label="Noise", alpha=0.6)
    else:
        plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], label=f"Cluster {label}")

plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segmentation using DBSCAN")
plt.legend()
plt.show()
