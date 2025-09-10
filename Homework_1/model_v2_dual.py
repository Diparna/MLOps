import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://raw.githubusercontent.com/Diparna/MLOps/refs/heads/main/Homework_1/sampregdata.csv")
data.head()

# Define feature (X) and target (y)
X = data[['x2','x4']]   
y = data['y']     

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Build and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

# Print results
print("Coefficient for X:", model.coef_[0])
print("Test R² for X:", r2)
print("Test RMSE for X:", rmse)
