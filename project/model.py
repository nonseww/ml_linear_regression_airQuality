import pandas as pd
import keras


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
    pass


# Generates a formatted string summary about th trained linear model
def model_info(feature_names, label_name, model_output):
    pass


# Core of the model: all functions are called from here
def run_experiment(df, feature_names, label_name, learning_rate, epochs, batch_size):
    pass
