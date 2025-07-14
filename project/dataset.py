import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# The function loads dataset
def load_dataset(name):
    dataset = pd.read_csv("../sources/{}".format(name), sep=';', decimal=',', usecols=lambda col: "Unnamed" not in col)
    training_df = dataset.loc[:, dataset.columns.tolist()]
    print('Read dataset completed successfully')
    print('Total numbers of rows: {0}\n\n'.format(len(training_df.index)))
    training_df = clear_df(training_df)
    return training_df


def show_missing_values(df, save_csv):
    missing_values = df.isnull().sum()
    print('Missing values: \n', missing_values)
    print('C6H6:', df['C6H6(GT)'].isna().sum())
    if save_csv:
        missing_values.to_csv('../data/missing_values.csv', header=['missing_values'])


# Function for analyze the missing values
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

    fig, ax = plt.subplots(figsize=(12, 4))
    missing_over_time.resample(period).sum().plot(
        ax=ax,
        title='Missing {} by {}'.format(col, {'h': 'Hour', 'D': 'Day', 'W': 'Week', 'M': 'Month'}[period]))
    if show_plot:
        plt.show()
    if save_plot:
        if path is None:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = f'../data/missing_values_plots/missing_{col}_{period}_{now}.png'
        fig.savefig(path)
    plt.close(fig)


# Drop rows where {column name} values is missing in long period
def drop_missing_blocks(df, name):
    df['missing'] = df[name].isnull().astype(int)
    df['missing_block'] = (df['missing'].diff(1) != 0).cumsum()
    block_sizes = df[df['missing'] == 1]['missing_block'].value_counts()
    large_blocks = block_sizes[block_sizes > 10].index
    df = df[~df['missing_block'].isin(large_blocks)].copy()
    df.drop(columns=['missing', 'missing_block'], inplace=True)
    return df


# Check multicollinearity via VIF (Variance Inflation Factor)
def check_multicollinearity(df):
    print(df.dtypes)
    numeric_df = df.drop(columns=['CO(GT)', 'Date', 'Time']).select_dtypes(include=['number']).copy()
    numeric_df.dropna(inplace=True)
    features = numeric_df.columns.tolist()
    cur_features = numeric_df[features].copy()
    cur_features = add_constant(cur_features)
    vif = pd.DataFrame()
    vif["Feature"] = cur_features.columns
    vif["VIF"] = [variance_inflation_factor(cur_features.values, i) for i in range(cur_features.shape[1])]
    vif.to_csv('../data/vif_results.txt', sep='\t', index=False)
    print('VIF result\n', vif)


# Explores df and returns features that should be used
def get_bad_features(df):
    check_multicollinearity(df)
    generate_correlation_matrix(df)
    # maybe some dimensionality reduction methods in the future
    return ['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'PT08.S4(NO2)', 'PT08.S5(O3)']


# Analyzes df and clears it from the missing values
def clear_df(df):
    df.replace(-200, np.nan, inplace=True)  # create the NaN instead of -200

    # get list of the missing values
    show_missing_values(df, True)

    # T, RH and AH are missed at all => drop it
    df.drop(columns=['T', 'RH', 'AH'], inplace=True)

    # NMHC(GT): 8557 out of 9471 rows => drop it, it's unusable
    df.drop(columns=['NMHC(GT)'], inplace=True)

    # work with NOx(GT): 1753 missings
    check_missing_values_by_time(df, 'NOx(GT)', 'D', False, False)

    # NOx(GT) has random missingness, that's okay, use interpolation
    df['NOx(GT)'] = df['NOx(GT)'].interpolate(limit=3)
    df = drop_missing_blocks(df, 'NOx(GT)')

    # delete the remaining missing ones
    df.dropna(subset=['NOx(GT)'], inplace=True)

    # check again
    check_missing_values_by_time(df, 'NOx(GT)', 'D', False, False)

    # work with NO2(GT): 1756 missings
    check_missing_values_by_time(df, 'NO2(GT)', 'D', False, False)

    # NO2(GT) has missing values from 0.001 to 4 missing valuer per day, but it's too noisy so use
    # interpolation carefully
    df['NO2(GT)'] = df['NO2(GT)'].interpolate(limit=2)

    # print(df['NO2(GT)'].isnull().sum()) # Debug: count of the missing values
    # it's only 24, so drop it
    df.dropna(subset=['NO2(GT)'], inplace=True)

    check_missing_values_by_time(df, 'NO2(GT)', 'D', False, False)

    # work with CO(GT): 526 missings
    check_missing_values_by_time(df, 'CO(GT)', 'D', False, False)

    # because CO(GT) is target, drop NaN rows
    df.dropna(subset=['CO(GT)'], inplace=True)

    # work with PT08.S1(CO): 329 missings
    check_missing_values_by_time(df, 'PT08.S1(CO)', 'D', False, False)
    df['PT08.S1(CO)'] = df['PT08.S1(CO)'].interpolate(limit=5, method='linear')
    df = drop_missing_blocks(df, 'PT08.S1(CO)')
    df.dropna(subset=['PT08.S1(CO)'], inplace=True)
    check_missing_values_by_time(df, 'PT08.S1(CO)', 'D', False, False)

    # works with C6H6(GT): 49 missings
    check_missing_values_by_time(df, 'C6H6(GT)', 'D', False, False)
    df['C6H6(GT)'] = df['C6H6(GT)'].interpolate(limit=5)
    check_missing_values_by_time(df, 'C6H6(GT)', 'D', False, False)

    # works with PT08.S2(NMHC): 49 missings
    check_missing_values_by_time(df, 'PT08.S2(NMHC)', 'D', False, False)
    df['PT08.S2(NMHC)'] = df['PT08.S2(NMHC)'].interpolate(limit=5)
    check_missing_values_by_time(df, 'PT08.S2(NMHC)', 'D', False, False)

    # works with PT08.S3(NOx): 49 missings
    check_missing_values_by_time(df, 'PT08.S3(NOx)', 'D', False, False)
    df['PT08.S3(NOx)'] = df['PT08.S3(NOx)'].interpolate(limit=5)
    check_missing_values_by_time(df, 'PT08.S3(NOx)', 'D', False, False)

    # works with PT08.S4(NO2): 49 missings
    check_missing_values_by_time(df, 'PT08.S4(NO2)', 'D', False, False)
    df['PT08.S4(NO2)'] = df['PT08.S4(NO2)'].interpolate(limit=5)
    check_missing_values_by_time(df, 'PT08.S4(NO2)', 'D', False, False)

    # works with PT08.S5(O3): 49 missings
    check_missing_values_by_time(df, 'PT08.S5(O3)', 'D', False, False)
    df['PT08.S5(O3)'] = df['PT08.S5(O3)'].interpolate(limit=5)
    check_missing_values_by_time(df, 'PT08.S5(O3)', 'D', False, False)

    # re-check missing values
    show_missing_values(df, False)

    # it's zero at all => good

    print('There are {} rows now'.format(len(df)))
    print('There are {} unique values'.format(df.nunique()))
    print('check {}'.format(df.select_dtypes(include='number').std()))

    # analyzing df for knowing which features should be used and drop
    df.drop(columns=get_bad_features(df), inplace=True)
    generate_correlation_matrix(df)
    return df


# Generate a correlation matrix
def generate_correlation_matrix(df):
    pd.set_option('display.max_columns', None)
    correlation_matrix = df.corr(numeric_only=True)
    print('Correlation matrix:\n', correlation_matrix)
