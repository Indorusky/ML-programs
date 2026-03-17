import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Load dataset
data = pd.read_csv("heart.csv")

X = data[["age", "sex", "cholesterol", "bp"]]
y = data["target"]

# Train model
model = GaussianNB()
model.fit(X, y)

print("\n--- Heart Disease Diagnosis ---")

age = int(input("Enter age: "))
sex = int(input("Enter sex (1 = Male, 0 = Female): "))
chol = int(input("Enter cholesterol level: "))
bp = int(input("Enter blood pressure: "))

# Create input DataFrame
input_data = pd.DataFrame(
    [[age, sex, chol, bp]],
    columns=["age", "sex", "cholesterol", "bp"]
)

prediction = model.predict(input_data)

print("\nResult:")
if prediction[0] == 1:
    print("⚠ Heart Disease Detected")
else:
    print("✅ No Heart Disease Detected")
