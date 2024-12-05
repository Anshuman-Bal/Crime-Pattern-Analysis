import pandas as pd
from sklearn.ensemble import IsolationForest

# Load your dataset
data = pd.read_csv("dataset.csv")

# Select numeric columns for simplicity
X = data.select_dtypes(include="number")

# Train the Isolation Forest model
isolation_forest = IsolationForest(contamination=0.05)  # Adjust contamination based on your dataset
isolation_forest.fit(X)

# Predict outliers
predictions = isolation_forest.predict(X)

# Identify outliers
outliers = data[predictions == -1]

# Display identified outliers
print("Identified Outliers:")
print(outliers)
