# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load your dataset (assuming it's stored in a variable named 'dataset')
dataset = pd.read_csv('dataset.csv')
# Extract features (X) and target variable (y)
X = dataset.drop(['Crime Head'], axis=1)  # Exclude the target variable
y = dataset['Crime Head']

# Encode categorical variables
label_encoder = LabelEncoder()
X['State/UTs'] = label_encoder.fit_transform(X['State/UTs'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the decision tree model
dt_model.fit(X_train, y_train)

# Plot the decision tree
"""
plt.figure(figsize=(15, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=y.unique())
plt.show()
"""

# Make predictions on the testing set
y_pred = dt_model.predict(X_test)

# Evaluate the model
accuracy = dt_model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': dt_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print('\nFeature Importance:')
print(feature_importance)