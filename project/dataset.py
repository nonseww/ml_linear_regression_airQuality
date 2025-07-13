import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# The function loads dataset
def load_dataset(name):
    dataset = pd.read_csv("../sources/{}".format(name), sep=';')
    training_df = dataset.loc[:, dataset.columns.tolist()]
    print('Read dataset completed successfully')
    print('Total numbers of rows: {0}\n\n'.format(len(training_df.index)))
    return training_df
