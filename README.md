# F1 Race Prediction System

A machine learning system for predicting Formula 1 race results using historical stint data and tire strategies.

This system downloads F1 race data from FastF1, processes stint strategies, trains a machine learning model, and predicts upcoming race results based on historical patterns.

## Project Overview

A clean, simple machine learning system for predicting Formula 1 race results using historical data and stint strategies.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Virtual environment support

### Step 1: Check Python Version
```bash
python3 --version
```

### Step 2: Navigate to Project Directory
```bash
cd "/Users/prabhtjosan/Documents/F1 prediction"
```

### Step 3: Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

## Features

- **Data Ingestion**: Downloads F1 race data from FastF1 API
- **Stint Analysis**: Processes tire strategies and pit stops
- **Machine Learning**: Trains RandomForest model on historical patterns
- **Position Prediction**: Predicts final race positions (1-20)
- **Current Drivers**: Uses 2025 F1 driver grid

## Quick Start

### 1. Setup Environment
```bash
# Navigate to project
cd "/Users/prabhtjosan/Documents/F1 prediction"

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download F1 Data
```bash
python src/ingest_all.py
```

### 3. Train Model
```bash
python src/train_positions.py
```

### 4. Generate Predictions
```bash
python src/predict_upcoming.py
```

## How to Run the Project

### Data Ingestion
- Downloads race data from 2022-2025 seasons
- Caches data locally for faster access
- Saves as `.parquet` files in `data/raw/`
- Skips already downloaded races

**Expected output:** Race files for each GP (may take 10-15 minutes first time)

### Model Training
Train the ML model using downloaded data:

```bash
python src/train_positions.py
```

**Expected output:**
```
INFO:__main__:Training with 42 features on 498 samples
INFO:__main__:Training MAE: 0.82
INFO:__main__:Test MAE: 2.19
INFO:__main__:Model saved as model.pkl
```

### Race Predictions
Generate predictions for upcoming races:

```bash
python src/predict_upcoming.py
```

**Expected output:**
```
SINGAPORE GRAND PRIX 2025 - RACE PREDICTION
==================================================
 1. VER [P1]
 2. LEC [P2]
 3. HAM [P3]
 4. RUS [POINTS]
 5. NOR [POINTS]
 6. PIA [POINTS]
 7. ALO [POINTS]
 8. STR [POINTS]
 9. GAS [POINTS]
10. OCO [POINTS]
11. ALB 
12. COL 
13. TSU 
14. LAW 
15. HUL 
16. MAG 
17. BOR 
18. ZHO 
19. BEA 
20. SAI 
```

## Project Structure

```
F1 prediction/
├── src/
│   ├── ingest_all.py         # Download F1 data
│   ├── train_positions.py    # Train ML model
│   └── predict_upcoming.py   # Generate predictions
├── data/
│   └── raw/                  # Race data (auto-created)
├── f1_cache/                 # FastF1 cache (auto-created)
├── requirements.txt          # Python dependencies
├── model.pkl                 # Trained model (auto-created)
├── feature_columns.pkl       # Model features (auto-created)
└── README.md                 # This file
```

## Command Reference

### 1. Data Ingestion
Download F1 race data from 2022-2025:

```bash
python src/ingest_all.py
```

### 2. Model Training
Train the prediction model:

```bash
python src/train_positions.py
```

### 3. Predict Upcoming Races
Generate predictions for future races:

```bash
python src/predict_upcoming.py
```

## Current F1 2025 Drivers

- **Red Bull**: VER, PER
- **Ferrari**: LEC, SAI
- **Mercedes**: HAM, RUS
- **McLaren**: NOR, PIA
- **Aston Martin**: STR, ALO
- **Alpine**: GAS, OCO
- **Williams**: ALB, COL
- **AlphaTauri**: TSU, LAW
- **Haas**: HUL, MAG
- **Sauber**: BOR, ZHO

## Troubleshooting

### Problem: "No race data found"
**Solution:**
1. Run data ingestion first: `python src/ingest_all.py`
2. Check `data/raw/` directory exists and contains `.parquet` files

### Problem: "Model not found"
**Solution:**
1. Train the model first: `python src/train_positions.py`
2. Check `model.pkl` file exists in project root

### Problem: FastF1 cache errors
**Solution:**
1. Delete cache directory: `rm -rf f1_cache/`
2. Re-run ingestion: `python src/ingest_all.py`

### Problem: Rate limit errors
**Solution:**
- Wait 1 hour before retrying (F1 API has 500 calls/hour limit)
- Cached data will be used automatically

## Understanding the Output

### Position Indicators
- [P1][P2][P3] = Podium positions
- [POINTS] = Points positions (4th-10th)

### Model Performance
- Training MAE: ~0.8 positions (very good)
- Test MAE: ~2.2 positions (reasonable for F1)
- Most important feature: Start position (71% importance)

## Development Tips

### Adding More Data
- Modify years in `ingest_all.py` to include more seasons
- Current: 2022-2025 (limited by API availability)

### Improving Model
- Try different algorithms in `train_positions.py`
- Add weather data, qualifying results
- Include track characteristics

### Custom Predictions
- Modify driver grid in `predict_upcoming.py`
- Change race name and date
- Add custom starting positions

## Next Steps

1. **More Data**: Add 2024-2025 races when available
2. **Better Features**: Include qualifying, weather, track data
3. **Advanced Models**: Try XGBoost, Neural Networks
4. **Live Updates**: Automated race weekend predictions
5. **Validation**: Backtest on historical races

## Model Performance

- **Training Accuracy**: Mean Absolute Error of 0.82 positions
- **Test Accuracy**: Mean Absolute Error of 2.19 positions
- **Key Features**: Start position (71%), lap times, stint strategies
- **Training Data**: 1,089 stint records from 2022-2023 F1 seasons

## Getting Started Tips

1. **First Run**: Start with `python src/ingest_all.py` to download data
2. **Development**: Use virtual environment to avoid conflicts
3. **Debugging**: Check logs for detailed error messages
4. **Performance**: Model training takes 1-2 minutes on modern hardware
5. **Data Size**: Expect ~50MB of race data after full ingestion