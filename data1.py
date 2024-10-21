import pandas as pd
import numpy as np

# Simulate historical sales data for a few product categories
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='W')

data = {
    'date': dates,
    'product_A': np.random.poisson(lam=20, size=len(dates)),
    'product_B': np.random.poisson(lam=30, size=len(dates)),
    'product_C': np.random.poisson(lam=15, size=len(dates)),
    'special_event': [1 if date.month == 12 else 0 for date in dates]  # Example for holiday season effect
}

sales_data = pd.DataFrame(data)
print(sales_data.head())
