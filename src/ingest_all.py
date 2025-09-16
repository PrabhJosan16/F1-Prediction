"""
F1 Data Ingestion Script
Downloads race data from FastF1 and saves as parquet files
"""

import fastf1
import polars as pl
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable FastF1 cache
fastf1.Cache.enable_cache('f1_cache')

def get_race_schedule(year):
    """Get race schedule for a given year"""
    try:
        schedule = fastf1.get_event_schedule(year)
        return schedule[schedule['EventFormat'] == 'conventional']
    except Exception as e:
        logger.error(f"Error getting schedule for {year}: {e}")
        return None

def process_race(year, round_num, event_name):
    """Process a single race and extract stint data"""
    try:
        logger.info(f"Processing {year} Round {round_num}: {event_name}")
        
        # Load race session
        race = fastf1.get_session(year, round_num, 'R')
        race.load()
        
        # Get stint data
        stint_data = []
        
        for driver in race.drivers:
            driver_laps = race.laps.pick_drivers(driver)
            
            if len(driver_laps) == 0:
                continue
                
            # Group by stint (tire compound changes)
            stints = driver_laps.pick_drivers(driver).groupby(['Compound']).agg({
                'LapNumber': ['min', 'max', 'count'],
                'LapTime': 'mean',
                'Position': ['first', 'last']
            }).reset_index()
            
            for _, stint in stints.iterrows():
                stint_data.append({
                    'year': year,
                    'round': round_num,
                    'event_name': event_name,
                    'driver': driver,
                    'compound': str(stint['Compound']),  # Convert to string
                    'stint_start_lap': stint[('LapNumber', 'min')],
                    'stint_end_lap': stint[('LapNumber', 'max')],
                    'stint_length': stint[('LapNumber', 'count')],
                    'avg_lap_time': stint[('LapTime', 'mean')].total_seconds() if pd.notna(stint[('LapTime', 'mean')]) else None,
                    'start_position': stint[('Position', 'first')],
                    'end_position': stint[('Position', 'last')]
                })
        
        return stint_data
        
    except Exception as e:
        logger.error(f"Error processing {year} Round {round_num}: {e}")
        return []

def ingest_year(year):
    """Ingest all races for a given year"""
    logger.info(f"Starting ingestion for {year}")
    
    schedule = get_race_schedule(year)
    if schedule is None:
        return
    
    all_stint_data = []
    
    for _, event in schedule.iterrows():
        if pd.isna(event['RoundNumber']):
            continue
            
        round_num = int(event['RoundNumber'])
        event_name = event['EventName']
        
        # Check if file already exists
        output_file = Path(f"data/raw/{year}_round_{round_num:02d}_{event_name.replace(' ', '_')}.parquet")
        if output_file.exists():
            logger.info(f"Skipping {event_name} - already exists")
            continue
        
        stint_data = process_race(year, round_num, event_name)
        
        if stint_data:
            # Convert to DataFrame and save
            df = pd.DataFrame(stint_data)
            df.to_parquet(output_file)
            logger.info(f"Saved {len(stint_data)} stint records for {event_name}")
            all_stint_data.extend(stint_data)
    
    logger.info(f"Completed {year}: {len(all_stint_data)} total stint records")

def main():
    """Main ingestion function"""
    # Create data directory
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    # Ingest data for each year
    years = [2022, 2023, 2024, 2025]
    
    for year in years:
        try:
            ingest_year(year)
        except Exception as e:
            logger.error(f"Failed to ingest {year}: {e}")
    
    logger.info("Data ingestion complete!")

if __name__ == "__main__":
    main()