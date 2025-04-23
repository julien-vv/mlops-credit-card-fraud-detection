# Import
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Read the file
def read_file(path):
    return pd.read_csv(path)

# Preprocess the dataset
def preprocessing_data(df):
    # Create the 'Hour' column
    df['Hour'] = (df['Time'] // 3600) % 24

    # Log the 'Amount column' before standardize
    df['Amount'] = np.log1p(df['Amount'])

    # Scale 'Amount' and 'Hour' column
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Hour_scaled'] = scaler.fit_transform(df[['Hour']])

    # Drop old columns and print the result
    df.drop(columns=['Amount', 'Hour', 'Time'], inplace=True)

    return df

# Split the dataset for train and test
def prep_for_training(df):
    X = df.drop(columns=['Class'], axis=1)
    y = df['Class']
    return train_test_split(X, y, test_size=0.2, random_state=42)
