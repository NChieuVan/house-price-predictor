import joblib # type: ignore
import pandas as pd # type: ignore
from datetime import datetime
from pathlib import Path
from schemas import HousePredictionRequest, LocationEnum, PredictionResponse, ConditionEnum

# Load model and preprocessor
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "trained"
MODEL_PATH = MODEL_DIR / "house_price_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor: {str(e)}")

def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    """
    Predict house price based on input features.
    """
    # Prepare input data
    input_data = pd.DataFrame([request.dict()])
    input_data['house_age'] = datetime.now().year - input_data['year_built']
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
    input_data['price_per_sqft'] = 0  # Dummy value for compatibility

    # Preprocess input data
    processed_features = preprocessor.transform(input_data)

    # Make prediction
    predicted_price = model.predict(processed_features)[0]

    # Convert numpy.float32 to Python float and round to 2 decimal places
    predicted_price = round(float(predicted_price), 2)

    # Confidence interval (10% range)
    confidence_interval = [predicted_price * 0.9, predicted_price * 1.1]

    # Convert confidence interval values to Python float and round to 2 decimal places
    confidence_interval = [round(float(value), 2) for value in confidence_interval]
    
    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_interval=confidence_interval,
        feautures_importance={},  # Placeholder for feature importance
        predcition_time=datetime.now().isoformat()
    )

def batch_predict(requests: list[HousePredictionRequest]) -> list[float]:
    """
    Perform batch predictions.
    """
    input_data = pd.DataFrame([req.dict() for req in requests])
    input_data['house_age'] = datetime.now().year - input_data['year_built']
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
    input_data['price_per_sqft'] = 0  # Dummy value for compatibility

    # Preprocess input data
    processed_features = preprocessor.transform(input_data)

    # Make predictions
    predictions = model.predict(processed_features)
    return predictions.tolist()

if __name__ == "__main__":
    # Example usage
    sample_request = HousePredictionRequest(
        sqft=2000,
        bedrooms=3,
        bathrooms=2,
        location=LocationEnum.suburb,
        year_built=1990,
        condition=ConditionEnum.good
    )
    prediction = predict_price(sample_request)
    print(prediction)

    sample_batch_request = [
        HousePredictionRequest(
            sqft=2000,
            bedrooms=3,
            bathrooms=2,
            location=LocationEnum.suburb,
            year_built=1990,
            condition=ConditionEnum.good
        ),
        HousePredictionRequest(
            sqft=1500,
            bedrooms=2,
            bathrooms=1,
            location=LocationEnum.downtown,
            year_built=1980,
            condition=ConditionEnum.fair
        )
    ]
    batch_predictions = batch_predict(sample_batch_request)
    print(batch_predictions)
    
    