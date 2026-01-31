import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay

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



def rename_uci_columns(df):
    """
    Renames the original UCI Bank Marketing dataset columns to clearer, 
    more descriptive names that are easier to understand and use in analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw UCI dataset.

    Returns
    -------
    pandas.DataFrame
        A dataframe with renamed columns.
    """

    rename_dict = {
        'age': 'age',
        'job': 'job_type',
        'marital': 'marital_status',
        'education': 'education_level',
        'default': 'credit_default',
        'balance': 'account_balance',
        'housing': 'housing_loan',
        'loan': 'personal_loan',
        'contact': 'contact_type',
        'day': 'contact_day',
        'month': 'contact_month',
        'duration': 'call_duration_sec',
        'campaign': 'num_contacts_current_campaign',
        'pdays': 'days_since_last_contact',
        'previous': 'num_previous_contacts',
        'poutcome': 'previous_outcome',
        'y': 'subscribed'
    }

    return df.rename(columns=rename_dict)



def create_age_groups(df, age_col='age'):
    """
    Creates categorical age groups (bins) from a numerical age column.
    Useful for segmentation, visualization, and behavioral analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing the age column.
    age_col : str
        The name of the numerical age column.

    Returns
    -------
    pandas.DataFrame
        A dataframe with a new column 'age_group_bin'.
    """

    df['age_group_bin'] = pd.cut(
        df[age_col],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    )

    return df

def create_campaign_groups(df, campaign_col='num_contacts_current_campaign'):
    """
    Groups the number of contacts made during the current campaign into 
    meaningful categories. Helps analyze diminishing returns and customer fatigue.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing the campaign column.
    campaign_col : str
        The name of the numerical campaign column.

    Returns
    -------
    pandas.DataFrame
        A dataframe with a new column 'campaign_group_bin'.
    """

    df['campaign_group'] = pd.cut(
        df[campaign_col],
        bins=[0, 1, 3, 5, 10, 100],
        labels=['1 contact', '2–3 contacts', '4–5 contacts', '6–10 contacts', '10+ contacts']
    )

    return df


def create_contact_missing_flag(df, contact_col='contact_type'):
    """
    Creates a binary flag indicating whether the contact_type field 
    was missing in the original dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing the contact column.
    contact_col : str
        The name of the contact column.

    Returns
    -------
    pandas.DataFrame
        A dataframe with a new column 'contact_missing_flag'.
    """

    df['contact_missing_flag'] = df[contact_col].isna().astype(int)
    return df




def clean_uci_dataset(df):
    """
    Applies all cleaning steps to the UCI dataset:
    - Renames columns
    - Creates age groups
    - Creates campaign groups
    - Creates missing contact flag

    Parameters
    ----------
    df : pandas.DataFrame
        The raw UCI dataset.

    Returns
    -------
    pandas.DataFrame
        A fully cleaned and enriched dataset ready for EDA and modeling.
    """

    df = rename_uci_columns(df)
    df = create_age_groups(df)
    df = create_campaign_groups(df)
    df = create_contact_missing_flag(df)

    return df



def detect_outliers_iqr(df, columns, method="flag"):
    """
    Detects outliers in numerical columns using the IQR (Interquartile Range) method.
    Can either flag, remove, or cap outliers depending on the selected method.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing numerical columns.
    columns : list
        List of numerical column names to check for outliers.
    method : str, optional
        How to handle outliers:
        - "flag": create a binary flag column for each variable
        - "remove": drop rows containing outliers
        - "cap": cap outliers to the IQR boundaries
        Default is "flag".

    Returns
    -------
    pandas.DataFrame
        A dataframe with outliers flagged, removed, or capped.
    """

    df = df.copy()

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if method == "flag":
            df[f"{col}_outlier_flag"] = (
                (df[col] < lower_bound) | (df[col] > upper_bound)
            ).astype(int)

        elif method == "remove":
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        elif method == "cap":
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        else:
            raise ValueError("method must be 'flag', 'remove', or 'cap'")

    return df







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
    


from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

def preprocess_data(X_train, X_test, categorical_cols, numerical_cols):
    """
    Preprocesses training and test data:
    - One-hot encodes categorical columns
    - Scales numerical columns
    - Concatenates both into final model-ready matrices
    - Returns fitted encoder and scaler for future use
    """

    # ----- 1. One-hot encoding -----
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    ohe.fit(X_train[categorical_cols])

    X_train_cat = ohe.transform(X_train[categorical_cols])
    X_test_cat = ohe.transform(X_test[categorical_cols])

    X_train_cat_df = pd.DataFrame(X_train_cat, 
                                  columns=ohe.get_feature_names_out(),
                                  index=X_train.index)

    X_test_cat_df = pd.DataFrame(X_test_cat, 
                                 columns=ohe.get_feature_names_out(),
                                 index=X_test.index)

    # ----- 2. Scaling numerical columns -----
    scaler = StandardScaler()
    scaler.fit(X_train[numerical_cols])

    X_train_num = scaler.transform(X_train[numerical_cols])
    X_test_num = scaler.transform(X_test[numerical_cols])

    X_train_num_df = pd.DataFrame(X_train_num,
                                  columns=scaler.get_feature_names_out(),
                                  index=X_train.index)

    X_test_num_df = pd.DataFrame(X_test_num,
                                 columns=scaler.get_feature_names_out(),
                                 index=X_test.index)

    # ----- 3. Concatenate -----
    X_train_full = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
    X_test_full = pd.concat([X_test_num_df, X_test_cat_df], axis=1)

    return X_train_full, X_test_full, ohe, scaler



def engineer_features_uci(df, target_col='subscribed'):
    """
    Full feature engineering pipeline for the UCI bank marketing dataset.

    Steps:
        - Encode target variable (yes/no → 1/0)
        - Identify categorical and numeric columns
        - One-hot encode categorical variables
        - Scale numeric variables (fit on train only)
        - Train/test split with stratification

    Returns:
        X_train, X_test, y_train, y_test
    """

    df = df.copy()

    # 1. Encode target variable
    df[target_col] = df[target_col].map({'no': 0, 'yes': 1})

    # 2. Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Identify column types (AFTER feature engineering)
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 4. One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 5. Train/test split (AFTER encoding)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Scale numeric variables (fit on train only)
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test

 

    return X_train, X_test, y_train, y_test


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




def evaluate_model(model, X_test, y_test, threshold=0.5, title="Model Evaluation"):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n📊 {title} (Threshold = {threshold}):")
    print("----------------------------------------")
    print(f"Accuracy      : {acc:.2f}")
    print(f"Precision     : {prec:.2f}")
    print(f"Recall        : {rec:.2f}")
    print(f"F1 Score      : {f1:.2f}")
    print(f"AUC (ROC)     : {auc:.2f}")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=["#3A7BD5", "#00d2ff"])





def clean_loan_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the loan dataset.

    Steps:
    - Standardise column names to snake_case
    - Fix over-split column names (loan_i_d → loan_id, d_t_i_ratio → dti_ratio)
    - Standardise string columns (lowercase, strip)
    - Ensure loan_id is string
    - Rename target column 'default' to 'loan_default'
    """

    df_clean = df.copy()

    # Convert column names to snake_case
    def camel_to_snake(name):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

    df_clean.columns = [camel_to_snake(col) for col in df_clean.columns]

    # Fix over-split column names
    rename_map = {
        'loan_i_d': 'loan_id',
        'd_t_i_ratio': 'dti_ratio'
    }
    df_clean = df_clean.rename(columns=rename_map)

    # Ensure loan_id is string
    if 'loan_id' in df_clean.columns:
        df_clean['loan_id'] = df_clean['loan_id'].astype(str)

    # Standardise string columns (exclude loan_id)
    str_cols = [
        col for col in df_clean.select_dtypes(include='object').columns
        if col != 'loan_id'
    ]

    for col in str_cols:
        df_clean[col] = df_clean[col].str.strip().str.lower()

    # Rename target column
    if 'default' in df_clean.columns:
        df_clean = df_clean.rename(columns={'default': 'loan_default'})
    else:
        raise ValueError("Target column 'default' not found in dataset.")

    print("Loan dataset cleaned. Shape:", df_clean.shape)
    return df_clean
