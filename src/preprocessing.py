import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path_list: list) -> pd.DataFrame:
    """
    Load data from multiple CSV files
    """

    df_list = []

    for path in path_list:
        df = pd.read_csv(path)
        df_list.append(df)

    return df_list

def merge_data(df1: pd.DataFrame, df2: pd.DataFrame, on: str) -> pd.DataFrame:
    """
    Merge multiple DataFrames into a single DataFrame
    """

    df1_copy = df1.copy()
    df2_copy = df2.copy()

    df1_copy.drop_duplicates(subset=on, inplace=True)
    df2_copy.drop_duplicates(subset=on, inplace=True)

    df1_copy[on] = df1_copy[on].astype(str).str.strip().str.split('-').str[0]
    df2_copy[on] = df2_copy[on].astype(str).str.strip().str.split('-').str[0]

    merged_df = pd.merge(df1_copy, df2_copy, on=on, how='inner')

    return merged_df

def change_dtypes(df: pd.DataFrame, str_cols: list, date_cols: list) -> pd.DataFrame:
    """
    Change data types of columns in a DataFrame, converting specified columns to string and date types. 
    Rest of the columns will be converted to numeric types.
    """
    df_copy = df.copy()
    for col in str_cols:
        df_copy[col] = df_copy[col].astype(str).str.strip()

    for col in date_cols:
        df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')

    for col in df_copy.columns:
        if col not in str_cols and col not in date_cols:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    return df_copy