from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model and target names
model = joblib.load("model.joblib")
target_names = joblib.load("target_names.joblib")

# Define the input format
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Iris Classifier API is running"}

@app.post("/predict")
def predict_species(iris: IrisFeatures):
    data = [[
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width
    ]]
    prediction = model.predict(data)[0]
    return {"predicted_species": prediction}
