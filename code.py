# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Step 2: Simulate historical sales data for different products
dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='W')  # Weekly data

# Randomly generate sales data for fruits, vegetables, frozen food
np.random.seed(42)
data = {
    'date': dates,
    'fruits': np.random.poisson(lam=50, size=len(dates)),  # Simulate sales (avg 50 units)
    'vegetables': np.random.poisson(lam=80, size=len(dates)),  # Simulate sales (avg 80 units)
    'frozen_food': np.random.poisson(lam=30, size=len(dates)),  # Simulate sales (avg 30 units)
    'special_event': [1 if date.month == 12 else 0 for date in dates]  # High demand during Dec (holidays)
}

# Step 3: Create a DataFrame with the simulated data
df = pd.DataFrame(data)

# Add a feature for seasonality (week of year)
df['week_of_year'] = df['date'].dt.isocalendar().week

# Step 4: Define Features (X) and Targets (y)
# X = input features (week_of_year, special_event), y = target variables (sales for each product)
X = df[['week_of_year', 'special_event']]
y = df[['fruits', 'vegetables', 'frozen_food']]

# Step 5: Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Random Forest model to predict demand
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 8: Evaluate the model's performance using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")  # A lower MAE means better accuracy

# Step 9: Plot actual vs predicted sales for a better understanding
plt.figure(figsize=(10, 6))
plt.plot(y_test['fruits'].values, label='Actual Fruit Sales', marker='o')
plt.plot(y_pred[:, 0], label='Predicted Fruit Sales', marker='x')
plt.title('Actual vs Predicted Demand for Fruits')
plt.xlabel('Test Samples')
plt.ylabel('Sales')
plt.legend()
plt.show()
