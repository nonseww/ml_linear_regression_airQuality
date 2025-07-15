import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler


# Creates and compiles a linear regression model
def build_model(learning_rate, num_features):
    inputs = keras.Input(shape=(num_features,))
    outputs = keras.layers.Dense(units=1)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
                  loss="mean_squared_error",
                  metrics=[keras.metrics.RootMeanSquaredError()])
    return model


# Trains the given model
def train_model(model, features, label, epochs, batch_size):
    history = model.fit(x=features,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs)
    trained_weight = model.get_weights()[0]  # weights matrix
    trained_bias = model.get_weights()[1]  # bias vector
    epochs = history.epoch
    hist = pd.DataFrame(history.history)  # converts the history dict into a pandas DataFrame
    rmse = hist["root_mean_squared_error"]
    return trained_weight, trained_bias, epochs, rmse


# Generates a formatted string summary about the trained linear model
def model_info(feature_names, label_name, model_output):
    weights = model_output[0]
    bias = model_output[1]
    nl = "\n"
    header = "-" * 80
    banner = header + nl + "|" + "MODEL INFO".center(78) + "|" + nl + header
    info = ""
    equation = label_name + " = "

    for index, feature in enumerate(feature_names):
        info += "Weight for feature[{}]: {:.3f}\n".format(feature, weights[index][0])
        equation += "{:.3f} * {} + ".format(weights[index][0], feature)

    info += "Bias: {:.3f}\n".format(bias[0])
    equation += "{:.3f}\n".format(bias[0])

    return banner + nl + info + nl + equation


def get_standardized_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)


# Core of the model: all functions are called from here
def run_training(df, feature_names, label_name, learning_rate, epochs, batch_size):
    print("INFO: start training the model with features={} and label={}\n".format(feature_names, label_name))
    num_features = len(feature_names)
    model = build_model(learning_rate, num_features)
    features = df.loc[:, feature_names].values
    features = get_standardized_features(features)
    label = df[label_name].values
    model_result = train_model(model, features, label, epochs, batch_size)

    print('\nSUCCESS: training complete\n')
    print('{}'.format(model_info(feature_names, label_name, model_result)))

    return model
