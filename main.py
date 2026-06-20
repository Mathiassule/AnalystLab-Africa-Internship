from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Price Prediction API",
    description="A simple API to predict house prices based on square footage and bedrooms.",
    version="1.0"
)

# 2. Load the trained model
model_filename = 'price_prediction_model.pkl'
try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise RuntimeError(f"Model file '{model_filename}' not found. Please run train.py first.")

# 3. Define the data schema for the incoming JSON request
class HouseFeatures(BaseModel):
    sqft: float
    bedrooms: int

# 4. Define the prediction endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        # Format the data into a 2D array for the scikit-learn model
        input_data = np.array([[features.sqft, features.bedrooms]])
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Return the result as JSON
        return {
            "sqft": features.sqft,
            "bedrooms": features.bedrooms,
            "predicted_price": round(prediction[0], 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Define a root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Price Prediction API. Navigate to /docs to test the model interactively."}