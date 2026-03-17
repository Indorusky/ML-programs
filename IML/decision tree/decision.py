import csv
import math
from collections import Counter

def load_data(filename):
    with open(filename, 'r') as f:
        # DictReader reads the first line as headers automatically
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames

def entropy(target_col):
    # Calculate frequency of each value in the target column
    elements_count = Counter(target_col)
    total = len(target_col)
    ent = 0
    for count in elements_count.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

def information_gain(data, feature, target):
    # Extract the target column values
    target_col = [row[target] for row in data]
    total_entropy = entropy(target_col)
    
    # Get unique values for the feature
    feature_values = [row[feature] for row in data]
    unique_values = set(feature_values)
    
    weighted_entropy = 0
    for v in unique_values:
        # Filter the data (subset)
        subset = [row for row in data if row[feature] == v]
        
        # Calculate entropy for this subset
        subset_target_col = [row[target] for row in subset]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset_target_col)
        
    return total_entropy - weighted_entropy

def id3(data, features, target):
    # Extract target values to check for purity
    target_col = [row[target] for row in data]
    unique_targets = set(target_col)
    
    # Base Case 1: If all target values are the same, return that value
    if len(unique_targets) == 1:
        return list(unique_targets)[0]
    
    # Base Case 2: If no features left, return the mode (most common value)
    if len(features) == 0:
        return Counter(target_col).most_common(1)[0][0]
    
    # Calculate Information Gain for all features
    gains = {f: information_gain(data, f, target) for f in features}
    
    # Select the best feature
    best_feature = max(gains, key=gains.get)
    
    tree = {best_feature: {}}
    
    # Remove the best feature from the list of features for the next recursion
    remaining_features = [f for f in features if f != best_feature]
    
    # Get unique values of the best feature to create branches
    feature_values = [row[best_feature] for row in data]
    unique_values = set(feature_values)
    
    for value in unique_values:
        # Create subset where best_feature == value
        subset = [row for row in data if row[best_feature] == value]
        tree[best_feature][value] = id3(subset, remaining_features, target)
        
    return tree

# --- Main Execution ---

# Load data using standard csv module
data, headers = load_data("tennis.csv")

# Assuming the last column is the target and the rest are features
target_column = headers[-1]
feature_columns = headers[:-1]

decision_tree = id3(data, feature_columns, target_column)

print("\nDecision Tree :\n")
print(decision_tree)
