from fastapi import FastAPI, Request, Depends, HTTPException, status, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import logging

# ===============================
# Logging Configuration
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ===============================
# Create FastAPI app
# ===============================
app = FastAPI(title="Titanic Survival Prediction API")

# ===============================
# Load trained model
# ===============================
with open("titanic_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
feature_columns = data["features"]

# ===============================
# Authentication Setup
# ===============================
API_KEY = "mysecretkey"
API_KEY_NAME = "x-api-key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    
    if api_key == API_KEY:
        return api_key
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key"
    )

# ===============================
# Request Schema with Validation
# ===============================
class PredictionRequest(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: int = Field(..., ge=0, le=1)
    Age: float = Field(..., ge=0)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked_Q: int = Field(..., ge=0, le=1)
    Embarked_S: int = Field(..., ge=0, le=1)

class PredictionResponse(BaseModel):
    prediction: int

# ===============================
# Validation Error Handler
# ===============================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"VALIDATION ERROR | Path: {request.url.path} | Errors: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

# ===============================
# Global Exception Handler
# ===============================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"ERROR | Path: {request.url.path} | Error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# ===============================
# Public Endpoint
# ===============================
@app.get("/")
def home():
    return {"message": "Titanic ML API is running"}

# ===============================
# Secured Prediction Endpoint
# ===============================
@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={401: {"description": "Unauthorized"}}
)
def predict(
    request: PredictionRequest,
    api_key: str = Security(verify_api_key)
):

    logger.info(f"REQUEST | Data: {request.dict()}")

    input_data = pd.DataFrame([request.dict()])
    input_data = input_data[feature_columns]

    prediction = model.predict(input_data)[0]

    logger.info(f"RESPONSE | Prediction: {prediction}")

    return {"prediction": int(prediction)}