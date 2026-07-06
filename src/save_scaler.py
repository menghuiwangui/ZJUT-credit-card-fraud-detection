"""Save the StandardScaler for Amount and Time features.
Run this once before using the Streamlit app.

Usage: python src/save_scaler.py
"""
import os
import joblib
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_data
import pandas as pd


def main():
    print('Loading raw data...')
    df = load_data('data/raw/creditcard.csv')
    
    # Fit scaler on Amount and Time (same as preprocess.py)
    scaler = StandardScaler()
    scaler.fit(df[['Amount', 'Time']])
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    scaler_path = os.path.join('models', 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    print(f'Scaler saved to: {scaler_path}')
    print(f'Scaler mean (Amount, Time): {scaler.mean_}')
    print(f'Scaler scale (Amount, Time): {scaler.scale_}')


if __name__ == '__main__':
    main()
