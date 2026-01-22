import numpy as np
import pandas as pd
import re

def explore_dataset(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print basic information about a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to explore.
    name : str, optional
        Name of the dataset (for display purposes).
    """
    print(f"=== {name} ===")
    print("\nShape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nInfo:")
    print(df.info())
    print("\nMissing values per column:\n", df.isna().sum()*100/len(df))
    print("\nFirst 5 rows:\n", df.head())



def clean_uci_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the UCI Bank Marketing dataset.

    Steps:
    - Replace 'unknown' with NaN in categorical columns
    - Convert target column 'y' to binary (yes=1, no=0)
    - Standardise string columns to lowercase (only if safe)

    Parameters
    ----------
    df : pd.DataFrame
        Raw UCI dataset

    Returns
    -------
    pd.DataFrame
        Cleaned dataset
    """
    df_clean = df.copy()

    # Replace 'unknown' with NaN in object columns
    obj_cols = df_clean.select_dtypes(include='object').columns
    df_clean[obj_cols] = df_clean[obj_cols].replace('unknown', np.nan)

    # Convert target column 'y' to binary
    df_clean['y'] = df_clean['y'].map({'yes': 1, 'no': 0})

    # Standardise string columns to lowercase safely
    for col in obj_cols:
        if df_clean[col].dropna().apply(lambda x: isinstance(x, str)).all():
            df_clean[col] = df_clean[col].str.lower()

    print("Cleaning complete. Shape:", df_clean.shape)
    return df_clean


def handle_missing_values_uci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values for the UCI Bank Marketing dataset.

    Steps:
    - Drop 'poutcome' column
    - Drop rows where 'job' or 'education' is missing
    - Impute remaining missing values with 'missing'

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df_mv = df.copy()

    # Drop 'poutcome' column
    if 'poutcome' in df_mv.columns:
        df_mv = df_mv.drop(columns=['poutcome'])

    # Drop rows missing job or education
    df_mv = df_mv.dropna(subset=['job', 'education'])

    # Impute remaining missing values
    df_mv = df_mv.fillna('missing')

    print("Missing-value handling complete. Shape:", df_mv.shape)
    return df_mv
  

def clean_loan_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the loan dataset.

    Steps:
    - Standardise column names to snake_case
    - Standardise string columns (lowercase, strip)
    - Ensure loan_id is string
    - Rename target column 'default' to 'loan_default'

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Cleaned loan dataset
    """
    df_clean = df.copy()

    # Convert column names to snake_case
    def camel_to_snake(name):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

    df_clean.columns = [camel_to_snake(col) for col in df_clean.columns]

    # Ensure loan_id is string
    if 'loan_id' in df_clean.columns:
        df_clean['loan_id'] = df_clean['loan_id'].astype(str)

    # Standardise string columns
    str_cols = df_clean.select_dtypes(include='object').columns
    for col in str_cols:
        df_clean[col] = df_clean[col].str.strip().str.lower()

    # Rename target column
    if 'default' in df_clean.columns:
        df_clean = df_clean.rename(columns={'default': 'loan_default'})

    print("Loan dataset cleaned. Shape:", df_clean.shape)
    return df_clean


