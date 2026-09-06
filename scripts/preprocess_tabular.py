"""
CardioRisk AI - Tabular Data Preprocessing
Downloads and processes the Kaggle Cardiovascular Disease dataset.
"""
import os
import pandas as pd
import numpy as np

def create_directories():
    os.makedirs("data/raw/tabular", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def download_kaggle_dataset():
    """
    You need to manually download the dataset from Kaggle first.
    URL: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
    
    Download 'cardio_train.csv' and place it in data/raw/tabular/
    """
    file_path = "data/raw/tabular/cardio_train.csv"
    if not os.path.exists(file_path):
        print(f"ERROR: Dataset not found at {file_path}")
        print("Please download it from Kaggle and place it in data/raw/tabular/")
        return None
    return pd.read_csv(file_path, sep=';')

def clean_tabular_data(df):
    """
    Clean and preprocess the tabular data.
    """
    # Age is in days, convert to years
    df['age_years'] = df['age'] // 365
    
    # Remove unrealistic values
    df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 250)]
    df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
    df = df[(df['height'] >= 100) & (df['height'] <= 220)]
    df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]
    
    # Create BMI
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df = df[(df['bmi'] >= 15) & (df['bmi'] <= 60)]
    
    # Create pulse pressure
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    # Select final features
    features = [
        'age_years', 'gender', 'height', 'weight', 'bmi',
        'ap_hi', 'ap_lo', 'pulse_pressure',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active',
        'cardio'
    ]
    
    df = df[features].copy()
    
    # Rename columns
    df.columns = [
        'age', 'gender', 'height_cm', 'weight_kg', 'bmi',
        'systolic_bp', 'diastolic_bp', 'pulse_pressure',
        'cholesterol', 'glucose', 'smoking', 'alcohol', 'physical_activity',
        'target'
    ]
    
    return df

def split_and_save(df, test_size=0.2, random_state=42):
    """
    Split data into train/test and save to processed folder.
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save processed data
    X_train.to_parquet("data/processed/X_train.parquet", index=False)
    X_test.to_parquet("data/processed/X_test.parquet", index=False)
    y_train.to_frame().to_parquet("data/processed/y_train.parquet", index=False)
    y_test.to_frame().to_parquet("data/processed/y_test.parquet", index=False)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Positive class ratio: {y_train.mean():.3f}")
    print("Data saved to data/processed/")

def main():
    create_directories()
    df = download_kaggle_dataset()
    
    if df is None:
        return
    
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    df_clean = clean_tabular_data(df)
    print(f"Cleaned data shape: {df_clean.shape}")
    
    split_and_save(df_clean)
    
    # Save feature descriptions
    feature_desc = pd.DataFrame({
        'feature': ['age', 'gender', 'height_cm', 'weight_kg', 'bmi',
                     'systolic_bp', 'diastolic_bp', 'pulse_pressure',
                     'cholesterol', 'glucose', 'smoking', 'alcohol', 
                     'physical_activity'],
        'description': [
            'Age in years',
            'Gender (1=female, 2=male)',
            'Height in cm',
            'Weight in kg',
            'Body Mass Index',
            'Systolic blood pressure (mmHg)',
            'Diastolic blood pressure (mmHg)',
            'Systolic - Diastolic',
            'Cholesterol (1=normal, 2=above normal, 3=well above)',
            'Glucose (1=normal, 2=above normal, 3=well above)',
            'Smoking (0=no, 1=yes)',
            'Alcohol (0=no, 1=yes)',
            'Physical activity (0=no, 1=yes)'
        ]
    })
    feature_desc.to_csv("data/processed/feature_descriptions.csv", index=False)
    print("Feature descriptions saved.")

if __name__ == "__main__":
    main()