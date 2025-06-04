import pandas as pd

def merge_dataframes_on_num_acc(dfs):
    """Merge une liste de dataframes sur la colonne 'Num_Acc'."""
    from functools import reduce
    return reduce(lambda left, right: pd.merge(left, right, on='Num_Acc'), dfs)

def print_unique_values(df, exclude=[]):
    """Affiche les valeurs uniques par colonne sauf celles exclues."""
    for col in df.columns:
        if col not in exclude:
            print(f"{col}: {df[col].unique()}")

def safe_replace_median(df, cols):
    """
    Remplace dans les colonnes cols les valeurs -1 par la médiane non -1.
    """
    for col in cols:
        median_val = df.loc[df[col] != -1, col].median()
        df[col] = df[col].replace(-1, median_val)
    return df
