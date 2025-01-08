from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pickle
import os
import yaml
from sentence_transformers import SentenceTransformer
import uvicorn
from pathlib import Path

# Define the input schema
class ProductInput(BaseModel):
    brand: Optional[str] = Field("", description="Brand of the product")
    title: Optional[str] = Field("", description="Title of the product")
    description: Optional[str] = Field("", description="Description of the product")
    feature: Optional[List[str]] = Field(default_factory=list, description="List of product features")

app = FastAPI()

config_path = Path(__file__).parent.parent / "configuration/config_inference.yaml"

if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found at {config_path}")

try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except yaml.YAMLError as e:
    raise RuntimeError(f"Error parsing configuration file: {e}")

# Access paths and other configurations with validation
try:
    MODEL_PATH = config["model"]["path"]
    EMBEDDER_NAME = config["embedder"]["name"]
    SERVER_HOST = config["server"].get("host", "127.0.0.1")  
    SERVER_PORT = config["server"].get("port", 8000)         
except KeyError as e:
    raise KeyError(f"Missing required configuration key: {e}")

# Validate paths
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the classifier
try:
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

# Load or instantiate the embedder
try:
    embedder = SentenceTransformer.load(EMBEDDER_NAME)
except Exception as e:
    try:
        embedder = SentenceTransformer(EMBEDDER_NAME)
    except Exception as inner_e:
        raise RuntimeError(f"Failed to load or instantiate the embedder: {inner_e}")

@app.post("/predict")
def predict_main_cat(product: ProductInput):
    """
    Predict the main category of a product based on its features.
    """
    try:
        # Construct text from input
        features_joined = " ".join(product.feature) if product.feature else ""
        text_input = f"{product.brand} {product.title} {product.description} {features_joined}"

        # Generate embedding
        X = embedder.encode([text_input])

        # Predict category
        pred = clf.predict(X)[0]

        return {"main_cat": pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
def root():
    """
    Health check endpoint.
    """
    return {"message": "API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)