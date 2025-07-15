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
    batch = build_batch(df, batch_size)
    predicted_values = model.predict_on_batch(x=batch.loc[:, features].values)
    data = {"PREDICTED_CO": [], "OBSERVED_CO": [], "L1_LOSS": [], features[0]: [], features[1]: [], features[2]: []}

    for i in range(batch_size):
        predicted = predicted_values[i][0]
        observed = batch.at[i, label]
        data["PREDICTED_CO"].append(format_currency(predicted))
        data["OBSERVED_CO"].append(format_currency(observed))
        data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
        data[features[0]].append(format_currency(batch.at[i, features[0]]))
        data[features[1]].append(format_currency(batch.at[i, features[1]]))
        data[features[2]].append(format_currency(batch.at[i, features[2]]))

    output_df = pd.DataFrame(data)
    return output_df


# Prints info about prediction
def show_predictions(output):
    pass
