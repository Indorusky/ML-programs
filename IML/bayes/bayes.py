# bayes.py
# Naive Bayes Classifier for Play Tennis Dataset
# Displays Actual vs Predicted values and Performance Metrics

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# 1. Load the dataset
# -----------------------------
data = pd.read_csv("tennisdata.csv")

# -----------------------------
# 2. Encode categorical values
# -----------------------------
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# -----------------------------
# 3. Split features and target
# -----------------------------
X = data.drop("PlayTennis", axis=1)
y = data["PlayTennis"]

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# 5. Create & Train Naive Bayes Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# 6. Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 7. Decode class labels (Yes / No)
# -----------------------------
actual_labels = label_encoders["PlayTennis"].inverse_transform(y_test)
predicted_labels = label_encoders["PlayTennis"].inverse_transform(y_pred)

# -----------------------------
# 8. Create Actual vs Predicted Table
# -----------------------------
results = pd.DataFrame({
    "Actual": actual_labels,
    "Predicted": predicted_labels
})

# -----------------------------
# 9. Evaluation Metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# -----------------------------
# 10. Output
# -----------------------------
print("Naive Bayes Classifier Performance")
print("----------------------------------")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1 Score  : {f1:.2f}")

print("\nActual vs Predicted Values")
print("--------------------------")
print(results)

print("\nConfusion Matrix")
print("----------------")
print(confusion_matrix(y_test, y_pred))
