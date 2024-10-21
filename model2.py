from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Feature engineering - add additional features like week of the year
sales_data['week_of_year'] = sales_data['date'].dt.isocalendar().week

# Define features and target
X = sales_data[['week_of_year', 'special_event']]  # Features (you can add more, like weather, holidays, etc.)
y = sales_data[['product_A', 'product_B', 'product_C']]  # Target: demand for products

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
