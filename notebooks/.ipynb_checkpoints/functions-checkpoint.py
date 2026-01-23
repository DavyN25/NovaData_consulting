import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
)

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
    print("\nPercentage of duplicated rows:\n", df.duplicated().mean() * 100)
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




def eda_uci_dataset(df, target_col='y'):
    """
    Perform basic EDA on the UCI bank marketing dataset with embedded explanations.
    
    What:
        - Univariate analysis of key numeric features
        - Target distribution
        - Selected bivariate relationships (numeric + categorical vs target)
        - Correlation matrix for numeric features
    
    Why:
        - To understand the data structure and distributions
        - To detect class imbalance and potential data quality issues
        - To identify variables that are likely to be predictive of the target
        - To inform feature engineering and model design in a business-relevant way
    """
    
    # 1. Target distribution
    print("\n[1] Target distribution")
    print("What: Distribution of the target variable.")
    print("Why: To check for class imbalance and understand baseline conversion rates.\n")
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df)
    plt.title('Target Distribution')
    plt.xlabel(f'{target_col}')
    plt.ylabel('Count')
    plt.show()
    
    # 2. Numeric distributions
    print("\n[2] Numeric feature distributions")
    print("What: Histograms of numeric variables.")
    print("Why: To detect skewness, outliers, and typical value ranges,")
    print("     which influence scaling choices and model robustness.\n")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols].hist(figsize=(12, 8), bins=30)
    plt.suptitle('Numeric Feature Distributions', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # 3. Subscription rate by job
    if 'job' in df.columns:
        print("\n[3] Subscription rate by job")
        print("What: Mean target value by job category.")
        print("Why: To identify professional segments with higher conversion rates,")
        print("     which is key for marketing segmentation and prioritization.\n")
        
        job_y = (df
                 .groupby('job')[target_col]
                 .mean()
                 .sort_values(ascending=False))
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=job_y.index, y=job_y.values)
        plt.title('Subscription Rate by Job')
        plt.xlabel('Job')
        plt.ylabel('Mean Subscription Rate')
        plt.xticks(rotation=45, ha='right')
        plt.show()
    
    # 4. Balance vs target
    if 'balance' in df.columns:
        print("\n[4] Balance vs target")
        print("What: Boxplot of account balance by target.")
        print("Why: To test whether higher balances are associated with higher subscription rates,")
        print("     which supports targeting higher-value clients.\n")
        
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=target_col, y='balance', data=df)
        plt.title('Balance by Target')
        plt.xlabel('Target')
        plt.ylabel('Balance')
        plt.ylim(df['balance'].quantile(0.01),
                 df['balance'].quantile(0.99))
        plt.show()
    
    # 5. Correlation matrix
    print("\n[5] Correlation matrix of numeric features")
    print("What: Correlation heatmap of numeric variables.")
    print("Why: To understand relationships between features and with the target,")
    print("     and to detect potential multicollinearity before modeling.\n")
    
    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()
    

"""


def engineer_features_uci(df, target_col='y'):
    """
    Feature engineering pipeline for the UCI bank marketing dataset.

    What:
        - Encode categorical variables
        - Scale numeric variables
        - Create optional interaction features
        - Split into train/test sets

    Why:
        - To prepare the dataset for machine learning models
        - To ensure consistent preprocessing
        - To support reproducibility and modularity

    Returns:
        X_train, X_test, y_train, y_test
    """

    # 1. Encode target variable
    df[target_col] = df[target_col].map({'no': 0, 'yes': 1})

    # 2. Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Identify column types
    categorical_cols = X.select_dtypes(include='object').columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # 4. One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 7. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    # 5. Scale numeric variables
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_encoded[numeric_cols] = scaler.transform(X_encoded[numeric_cols])

 

    return X_train, X_test, y_train, y_test
"""


def classification_diagnostic_plot(model, X, y, title="Classification Diagnostics"):
    """
    Generates confusion matrix, ROC curve, and key classification metrics.
    Works for binary classification models with predict_proba().
    """

    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)

    # Plot setup
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=axes[0], cmap="Greens")
    axes[0].set_title("Confusion Matrix")

    # ROC curve
    RocCurveDisplay.from_predictions(y, y_proba, ax=axes[1])
    axes[1].set_title("ROC Curve")

    # Metrics box
    textstr = (
        f"Accuracy: {acc:.2f}\n"
        f"Precision: {prec:.2f}\n"
        f"Recall: {rec:.2f}\n"
        f"F1 Score: {f1:.2f}\n"
        f"AUC: {auc:.2f}"
    )

    axes[1].text(
        0.6, 0.3,
        textstr,
        transform=axes[1].transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
