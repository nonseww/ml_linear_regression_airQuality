import keras


# Saves trained model into file
def save_model(model, name):
    model.save("../trained_models/{}".format(name))


# Loads the trained mode
def load_model(name):
    loaded_model = keras.models.load_model("../trained_models/{}".format(name))
    print("Model was loaded successfully\n")
    return loaded_model
