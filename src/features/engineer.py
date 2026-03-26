import pandas as pd # type: ignore
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
import joblib # type: ignore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature-engineer')

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing ones."""
    logger.info("Creating new features")
    
    df_features = df.copy()
    
    # Create a feature for the age of the house
    df_features['house_age'] = datetime.now().year - df_features['year_built']
    logger.info("Created feature: house_age")

    # Create a feature for price per square foot
    df_features['price_per_sqft'] = df_features['price'] / df_features['sqft']
    logger.info("Created feature: price_per_sqft")

    # Bedrooms to bathrooms ratio
    df_features['bed_bath_ratio'] = df_features['bedrooms'] / (df_features['bathrooms'] + 1)
    # Handle division by zero if bathrooms is zero
    df_features['bed_bath_ratio'] = df_features['bed_bath_ratio'].replace([np.inf, -np.inf], np.nan)
    df_features['bed_bath_ratio'] = df_features['bed_bath_ratio'].fillna(0)
    logger.info("Created feature: bed_bath_ratio")

    # Do not one-hot encode categorical features here, as it will be handled in the pipeline
    return df_features
    
def create_preprocessor():
    """Create a preprocessing pipeline."""
    logger.info("Creating preprocessing pipeline")
   
   # Define feature groups
    categorical_features = ['location', 'condition']
    numerical_features = ['sqft', 'bedrooms', 'bathrooms', 'house_age', 'price_per_sqft', 'bed_bath_ratio']


    # Preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def run_feature_engineering(input_file: str, output_file: str,processed_file: str):
    """Full feature engineering pipeline."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(processed_file).parent.mkdir(parents=True, exist_ok=True)

    # Load cleaned data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded cleaned data from {input_file}")

    # Create new features
    df_features = create_features(df)
    logger.info(f"Created new features with shape: {df_features.shape}")

    # Create add fit the preprocessor
    preprocessor = create_preprocessor()

    X = df_features.drop(columns=['price'],errors='ignore')
    y = df_features['price'] if 'price' in df_features.columns else None

    X_transformed = preprocessor.fit_transform(X)
    logger.info("Applied preprocessing pipeline to features")

    # Save the preprocessor for future use
    joblib.dump(preprocessor, processed_file)
    logger.info(f"Saved preprocessor to {processed_file}")

    # Save fully preprocessed data
    df_transformed = pd.DataFrame(X_transformed)
    if y is not None:
        df_transformed['price'] = y.values
    df_transformed.to_csv(output_file, index=False)
    logger.info(f"Saved preprocessed data to {output_file}")    

    return df_transformed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run feature engineering pipeline')
    parser.add_argument('--input', required=True,type=str ,help='Path to cleaned input data')
    parser.add_argument('--output',required=True, type=str, help='Path to save preprocessed data')
    parser.add_argument('--processor',required=True, type=str, help='Path to save preprocessor object')
    args = parser.parse_args()
    run_feature_engineering(args.input, args.output, args.processor)
