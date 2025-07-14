import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# The function loads dataset
def load_dataset(name):
    dataset = pd.read_csv("../sources/{}".format(name), sep=';', usecols=lambda col: "Unnamed" not in col)
    training_df = dataset.loc[:, dataset.columns.tolist()]
    print('Read dataset completed successfully')
    print('Total numbers of rows: {0}\n\n'.format(len(training_df.index)))
    training_df = clear_df(training_df)
    return training_df


# Function for analyze the missing values by using graphic
def check_missing_values_by_time(df, col, period, show_plot, save_plot, path=None):
    datetime_index = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H.%M.%S',
        errors='coerce')
    valid_mask = ~datetime_index.isna()
    df_valid = df.loc[valid_mask]
    datetime_index = datetime_index[valid_mask]
    missing_over_time = df_valid[col].isnull().astype(int)
    missing_over_time.index = datetime_index
    missing_over_time.resample(period).sum().plot(
        title='Missing {} by {}'.format(col, {'h': 'Hour', 'D': 'Day', 'W': 'Week', 'M': 'Month'}[period]),
        figsize=(12, 4))
    if show_plot:
        plt.show()
    if save_plot:
        save_path = path if path is not None else f'../data/missing_{col}_{period}.png'
        plt.savefig(save_path)
        plt.close()


# Drop rows where {column name} values is missing in long period
def drop_missing_blocks(df, name):
    df['missing'] = df[name].isnull().astype(int)
    df['missing_block'] = (df['missing'].diff(1) != 0).cumsum()
    block_sizes = df[df['missing'] == 1]['missing_block'].value_counts()
    large_blocks = block_sizes[block_sizes > 10].index
    df = df[~df['missing_block'].isin(large_blocks)].copy()
    df.drop(columns=['missing', 'missing_block'], inplace=True)
    return df


# Analyzing df and cleat it from the missing values
def clear_df(df):
    df.replace(-200, np.nan, inplace=True)  # create the NaN instead of -200

    # print(df.isnull().sum())  # Debug: check the count of the missing values
    # NMHC(GT): 8557 out of 9471 rows => drop it, it's unusable
    df.drop(columns=['NMHC(GT)'], inplace=True)

    # works with NOx(GT): 1753 missings
    check_missing_values_by_time(df, 'NOx(GT)', 'D', False, False)

    # NOx(GT) has random missingness, that's okay, use interpolation
    df['NOx(GT)'] = df['NOx(GT)'].interpolate(limit=3)
    df = drop_missing_blocks(df, 'NOx(GT)')

    # delete the remaining missing ones
    df.dropna(subset=['NOx(GT)'], inplace=True)

    # check again
    check_missing_values_by_time(df, 'NOx(GT)', 'D', False, True)

    # works with NO2(GT): 1756 missings
    check_missing_values_by_time(df, 'NO2(GT)', 'D', True, True)

    # NO2(GT) has missing values from 0.001 to 4 missing valuer per day, but it's too noisy so use
    # interpolation carefully
    df['NO2(GT)'] = df['NO2(GT)'].interpolate(limit=2)

    # print(df['NO2(GT)'].isnull().sum()) # Debug: count of the missing values
    # it's only 24, so drop it
    df.dropna(subset=['NO2(GT)'], inplace=True)

    check_missing_values_by_time(df, 'NO2(GT)', 'D', True, False)
    return df


# Generate a correlation matrix
def generate_correlation_matrix(df):
    pd.set_option('display.max_columns', None)
    correlation_matrix = df.corr(numeric_only=True)
    print(correlation_matrix)
