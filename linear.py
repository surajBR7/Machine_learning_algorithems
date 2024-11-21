import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Step 1: Load the dataset
df = pd.read_csv('./datasets/train.csv')

# Step 2: Define your predictors (X) and target (y)
df = df.dropna(subset=['y'])

# Redefine predictors (X) and target (y)
X = df.drop(columns=['y'])  # Drop target variable from the dataset
y = df['y']  # Target variable

# Step 3: Create a pipeline with an imputer and Linear Regression
imputer = SimpleImputer(strategy='mean')
pipeline = Pipeline(steps=[('imputer', imputer), ('model', LinearRegression())])

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model using the pipeline
pipeline.fit(X_train, y_train)

# Step 6: Make predictions on the test data
predictions = pipeline.predict(X_test)

# Step 7: Evaluate the model
r2_train = metrics.r2_score(y_train, pipeline.predict(X_train))
print(f'R² Value (Training): {r2_train:.4f}')

mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
accuracy = 100 - mape
print(f'Accuracy (Test Data): {accuracy:.2f}%')

# Optional: Create a DataFrame with actual and predicted values
TestingDataResults = pd.DataFrame(data=X_test, columns=X.columns)
TestingDataResults['Actual'] = y_test
TestingDataResults['Predicted'] = predictions
print(TestingDataResults.head())

# Step 8: Visualize Actual vs Predicted values for a specific feature
# For plotting, select the first feature in X_test
plt.figure(figsize=(10, 6))
plt.scatter(X_test.iloc[:, 0], y_test, color='red', label='Actual values')
plt.plot(X_test.iloc[:, 0], predictions, color='blue', linewidth=2, label='Predicted values')

plt.title('Actual vs Predicted Values')
plt.xlabel('First Feature of X_test')
plt.ylabel('Y (Target)')
plt.legend()
plt.show()
