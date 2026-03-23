import pandas as pd # type: ignore
import numpy as np
import logging
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature-engineer')

def create_features(df):
    """Create new features from existing ones."""
    logger.info("Creating new features")
    
    df_features = df.copy()
    
    # Create a feature for the age of the house
    df_features['house_age'] = datetime.now().year - df['year_built']
    
    # Create a feature for the total number of rooms
    df_features['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Create a feature for the price per square foot
    df_features['price_per_sqft'] = df['price'] / df['sqft_living']
    
    return df