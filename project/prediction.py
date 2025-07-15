import numpy as np
import pandas as pd


# Formats numbers
def format_currency(x):
    return "${:.2f}".format(x)


# Chooses a random group of data from the df
def build_batch(df, batch_size):
    batch = df.sample(n=batch_size).copy()
    batch.set_index(np.arange(batch_size), inplace=True)
    return batch


# Runs trained model on a random batch of data
def predict_fare(model, df, features, label, batch_size=50):
    pass


# Prints info about prediction
def show_predictions(output):
    pass
