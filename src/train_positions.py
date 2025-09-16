"""
F1 Position Prediction Model Training
Simple approach focusing on stint strategies and basic features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_race_data():
    """Load all race data from parquet files"""
    data_dir = Path("data/raw")
    all_data = []
    
    for file in data_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(file)
            all_data.append(df)
            logger.info(f"Loaded {len(df)} records from {file.name}")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not all_data:
        raise ValueError("No race data found. Run ingest_all.py first.")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total records loaded: {len(combined_df)}")
    return combined_df

def create_features(df):
    """Create features for machine learning"""
    logger.info("Creating features...")
    
    # Basic stint features
    df['stint_avg_lap_time_seconds'] = df['avg_lap_time']
    df['stint_length_normalized'] = df['stint_length'] / df['stint_length'].max()
    
    # Compound encoding (one-hot)
    compound_dummies = pd.get_dummies(df['compound'], prefix='compound')
    df = pd.concat([df, compound_dummies], axis=1)
    
    # Position change during stint
    df['position_change'] = df['end_position'] - df['start_position']
    
    # Driver encoding (limit to most common drivers)
    top_drivers = df['driver'].value_counts().head(20).index
    df['driver_encoded'] = df['driver'].apply(lambda x: x if x in top_drivers else 'OTHER')
    driver_dummies = pd.get_dummies(df['driver_encoded'], prefix='driver')
    df = pd.concat([df, driver_dummies], axis=1)
    
    # Year feature for temporal trends
    df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    
    logger.info(f"Features created. Shape: {df.shape}")
    return df

def prepare_training_data(df):
    """Prepare features and target for training"""
    # Target: final race position (simplified - using end_position of last stint per driver per race)
    race_results = df.groupby(['year', 'round', 'driver']).agg({
        'end_position': 'last',
        'stint_length': 'sum',
        'position_change': 'sum'
    }).reset_index()
    
    # Features: aggregate stint data per driver per race
    features = df.groupby(['year', 'round', 'driver']).agg({
        'stint_avg_lap_time_seconds': 'mean',
        'stint_length_normalized': 'sum',
        'start_position': 'first',
        'year_normalized': 'first'
    }).reset_index()
    
    # Add compound usage (count of each compound)
    compound_cols = [col for col in df.columns if col.startswith('compound_')]
    for col in compound_cols:
        compound_usage = df.groupby(['year', 'round', 'driver'])[col].sum().reset_index()
        features = features.merge(compound_usage, on=['year', 'round', 'driver'], how='left')
    
    # Add driver encoding
    driver_cols = [col for col in df.columns if col.startswith('driver_')]
    for col in driver_cols:
        driver_usage = df.groupby(['year', 'round', 'driver'])[col].max().reset_index()
        features = features.merge(driver_usage, on=['year', 'round', 'driver'], how='left')
    
    # Merge features with target
    training_data = features.merge(race_results[['year', 'round', 'driver', 'end_position']], 
                                 on=['year', 'round', 'driver'])
    
    # Fill NaN values
    training_data = training_data.fillna(0)
    
    # Ensure all feature columns are numeric
    feature_cols = [col for col in training_data.columns 
                   if col not in ['year', 'round', 'driver', 'end_position']]
    for col in feature_cols:
        training_data[col] = pd.to_numeric(training_data[col], errors='coerce').fillna(0)
    
    logger.info(f"Training data prepared. Shape: {training_data.shape}")
    return training_data

def train_model(training_data):
    """Train the position prediction model"""
    logger.info("Training model...")
    
    # Select feature columns (exclude identifiers and target)
    feature_cols = [col for col in training_data.columns 
                   if col not in ['year', 'round', 'driver', 'end_position']]
    
    X = training_data[feature_cols]
    y = training_data['end_position']
    
    logger.info(f"Training with {len(feature_cols)} features on {len(X)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    logger.info(f"Training MAE: {train_mae:.2f}")
    logger.info(f"Test MAE: {test_mae:.2f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 most important features:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Save model and feature columns
    joblib.dump(model, 'model.pkl')
    joblib.dump(feature_cols, 'feature_columns.pkl')
    
    logger.info("Model saved as model.pkl")
    return model, feature_cols

def main():
    """Main training function"""
    try:
        # Load data
        df = load_race_data()
        
        # Create features
        df = create_features(df)
        
        # Prepare training data
        training_data = prepare_training_data(df)
        
        # Train model
        model, feature_cols = train_model(training_data)
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()