import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("missing_data.csv")

# Display original data
print("Original Data:")
print(data)

# Create encoder
encoder = LabelEncoder()

# Apply encoding to categorical columns
for col in data.columns:
    data[col] = encoder.fit_transform(data[col].astype(str))

# Display encoded data
print("\nEncoded Data:")
print(data)