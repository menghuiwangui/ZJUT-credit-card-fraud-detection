import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(csv_path):
    """
    Load credit card fraud dataset and perform basic analysis.
    
    Parameters:
    csv_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Loaded dataframe
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Print dataset information
    print(f"Data shape: {df.shape}")
    
    # Missing values statistics
    missing_stats = df.isnull().sum()
    print("\nMissing values count:")
    print(missing_stats[missing_stats > 0])
    if missing_stats.sum() == 0:
        print("No missing values found in the dataset.")
    
    # Class distribution
    class_counts = df['Class'].value_counts()
    normal_count = class_counts.get(0, 0)
    fraud_count = class_counts.get(1, 0)
    total = len(df)
    
    print("\nClass distribution:")
    print(f"Normal transactions: {normal_count} ({normal_count/total:.2%})")
    print(f"Fraudulent transactions: {fraud_count} ({fraud_count/total:.2%})")
    
    return df

def eda_amount_time(df):
    """
    Perform exploratory data analysis on Amount and Time features.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    """
    # Set up the plotting style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Amount histogram
    sns.histplot(data=df, x='Amount', ax=axes[0], bins=50)
    axes[0].set_title('Distribution of Transaction Amount')
    
    # Time distribution
    sns.histplot(data=df, x='Time', ax=axes[1], bins=50)
    axes[1].set_title('Distribution of Transaction Time')
    
    # Class count bar plot
    sns.countplot(data=df, x='Class', ax=axes[2])
    axes[2].set_title('Count of Normal vs Fraudulent Transactions')
    axes[2].set_xlabel('Class (0=Normal, 1=Fraud)')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    # Example usage
    df = load_data("data/raw/creditcard.csv")
    eda_amount_time(df)
