"""
F1 Race Prediction Script
Predicts race results for upcoming events
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Current F1 2025 driver lineup
CURRENT_DRIVERS = [
    'VER', 'LEC', 'HAM', 'RUS', 'NOR', 'PIA', 'ALO', 'STR',
    'GAS', 'OCO', 'ALB', 'COL', 'TSU', 'LAW', 'HUL', 'MAG',
    'BOR', 'ZHO', 'SAI', 'BEA'
]

def load_model():
    """Load the trained model and feature columns"""
    try:
        model = joblib.load('model.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        logger.info("Model loaded successfully")
        return model, feature_cols
    except FileNotFoundError:
        raise FileNotFoundError("Model not found. Run train_positions.py first.")

def create_prediction_features(drivers):
    """Create features for prediction based on typical stint strategies"""
    predictions = []
    
    # Simulate typical race strategy features
    for i, driver in enumerate(drivers):
        # Starting position (randomized for demo - in reality would come from qualifying)
        start_pos = i + 1
        
        # Typical stint characteristics (simplified)
        features = {
            'stint_avg_lap_time_seconds': 90 + np.random.normal(0, 2),  # ~90s lap time
            'stint_length_normalized': 0.5 + np.random.normal(0, 0.1),  # Normalized stint length
            'start_position': start_pos,
            'year_normalized': 1.0,  # Current year (2025)
        }
        
        # Compound usage (simplified - assume typical 2-stop strategy)
        features['compound_HARD'] = 0
        features['compound_MEDIUM'] = 1
        features['compound_SOFT'] = 1
        
        # Driver encoding (set to 1 for current driver, 0 for others)
        for d in ['VER', 'LEC', 'HAM', 'RUS', 'NOR', 'PIA', 'ALO', 'STR', 'GAS', 'OCO']:
            features[f'driver_{d}'] = 1 if driver == d else 0
        features['driver_OTHER'] = 1 if driver not in ['VER', 'LEC', 'HAM', 'RUS', 'NOR', 'PIA', 'ALO', 'STR', 'GAS', 'OCO'] else 0
        
        features['driver'] = driver
        predictions.append(features)
    
    return pd.DataFrame(predictions)

def predict_race(race_name="Upcoming Race", drivers=None):
    """Predict race results"""
    if drivers is None:
        drivers = CURRENT_DRIVERS
    
    logger.info(f"Predicting results for: {race_name}")
    
    # Load model
    model, feature_cols = load_model()
    
    # Create prediction features
    pred_df = create_prediction_features(drivers)
    
    # Ensure all required features are present
    for col in feature_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0
    
    # Select only the features used in training
    X_pred = pred_df[feature_cols]
    
    # Make predictions
    predictions = model.predict(X_pred)
    
    # Create results dataframe
    results = pd.DataFrame({
        'driver': drivers,
        'predicted_position': predictions
    })
    
    # Sort by predicted position
    results = results.sort_values('predicted_position').reset_index(drop=True)
    results['position'] = range(1, len(results) + 1)
    
    # Display results
    print(f"\n{race_name.upper()} - RACE PREDICTION")
    print("=" * 50)
    
    for _, row in results.iterrows():
        pos = int(row['position'])
        driver = row['driver']
        
        # Add position indicators
        if pos == 1:
            indicator = "[P1]"
        elif pos == 2:
            indicator = "[P2]"
        elif pos == 3:
            indicator = "[P3]"
        elif pos <= 10:
            indicator = "[POINTS]"
        else:
            indicator = ""
        
        print(f"{pos:2d}. {driver} {indicator}")
    
    return results

def main():
    """Main prediction function"""
    try:
        # Predict upcoming race
        results = predict_race("Azerbaijan Grand Prix 2025")
        
        logger.info("Prediction complete!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

if __name__ == "__main__":
    main()