from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess(df):
    """
    Preprocess the credit card fraud dataset.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with 'Class' column
    
    Returns:
    tuple: X_train_res, X_test, y_train_res, y_test
    """
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Initialize scaler for Amount and Time columns
    scaler = StandardScaler()
    
    # Fit scaler on training data only for Amount and Time columns
    X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    return X_train_res, X_test, y_train_res, y_test
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess(df):
    """
    Preprocess the credit card fraud dataset.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with 'Class' column
    
    Returns:
    tuple: X_train_res, X_test, y_train_res, y_test
    """
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split the data first to ensure no data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Initialize and fit scaler on training data only
    scaler = StandardScaler()
    X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
    
    # Transform test data using the same scaler
    X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])
    
    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    return X_train_res, X_test, y_train_res, y_test
