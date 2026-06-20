House Price Prediction APIThis project is a simple Machine Learning pipeline that trains a Linear Regression model to predict house prices and deploys it locally using FastAPI.Project Structuretrain.py: A Python script that generates dummy real estate data, trains a scikit-learn Linear Regression model, and saves it as a serialized .pkl file.main.py: The FastAPI application that loads the trained model and serves a prediction endpoint.requirements.txt: The list of Python dependencies required to run the project.api_docs.pdf / api_docs.tex: Detailed API documentation containing input schemas, example requests, and responses.FeaturesMachine Learning: Utilizes scikit-learn for quick and efficient linear regression modeling.RESTful API: Built with FastAPI for high performance and automatic interactive documentation.Data Validation: Uses pydantic to ensure incoming API requests contain the correct data types.PrerequisitesPython 3.7 or higher installed on your system.Setup and Installation1. Clone the repository or navigate to the project directory:cd path/to/your/project
2. Install the required dependencies:pip install -r requirements.txt
UsageStep 1: Train the ModelBefore running the API, you must train the model and generate the price_prediction_model.pkl file.python train.py
You should see a success message indicating the model has been saved.Step 2: Start the API ServerLaunch the FastAPI server using Uvicorn. The --reload flag allows the server to automatically restart if you make code changes.uvicorn main:app --reload
Step 3: Test the APIOnce the server is running, you can interact with it via your browser:Root Endpoint: Navigate to http://127.0.0.1:8000/ for a simple welcome message.Interactive Documentation (Swagger UI): Navigate to http://127.0.0.1:8000/docs. Here you can test the /predict endpoint interactively without writing any frontend code or using Postman.API EndpointsPOST /predictPredicts the price of a house based on its square footage and number of bedrooms.Request Body Example:{
  "sqft": 2200,
  "bedrooms": 4
}
Response Example:{
  "sqft": 2200.0,
  "bedrooms": 4,
  "predicted_price": 319285.71
}
