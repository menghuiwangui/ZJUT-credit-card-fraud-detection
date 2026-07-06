import os
import pandas as pd
import kagglehub

# Define paths
RAW_DATA_DIR = "data/raw"
CREDITCARD_CSV = os.path.join(RAW_DATA_DIR, "creditcard.csv")

def main():
    # Create raw data directory if it doesn't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Skip download if file already exists
    if os.path.exists(CREDITCARD_CSV):
        print(f"File {CREDITCARD_CSV} already exists. Skipping download.")
        df = pd.read_csv(CREDITCARD_CSV)
    else:
        print("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        # The dataset contains only one file: creditcard.csv
        source_file = os.path.join(path, "creditcard.csv")
        df = pd.read_csv(source_file)
        df.to_csv(CREDITCARD_CSV, index=False)
        print(f"Dataset downloaded and saved to {CREDITCARD_CSV}")
    
    # Print dataset information
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nClass distribution:")
    print(df['Class'].value_counts())

if __name__ == "__main__":
    main()
