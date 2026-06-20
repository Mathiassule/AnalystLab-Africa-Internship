import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# 1. Generate some dummy data for house prices
# Features: Size (sqft) and Number of Bedrooms
data = {
    'sqft': [1200, 1500, 1800, 2200, 2500, 2800, 3100, 3500],
    'bedrooms': [2, 3, 3, 4, 4, 4, 5, 5],
    'price': [150000, 200000, 250000, 320000, 360000, 400000, 460000, 500000]
}

df = pd.DataFrame(data)

# 2. Split into features (X) and target (y)
X = df[['sqft', 'bedrooms']]
y = df['price']

# 3. Initialize and train the model
print("Training the Linear Regression model...")
model = LinearRegression()
model.fit(X, y)

# 4. Save the trained model as a .pkl file
model_filename = 'price_prediction_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model trained successfully and saved as '{model_filename}'")