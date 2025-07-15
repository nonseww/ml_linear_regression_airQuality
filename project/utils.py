from sklearn.preprocessing import StandardScaler
import joblib


# Standardize features (scaling)
def get_standardized_features(features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    joblib.dump(scaler, '../data/scaler.save')
    return scaled


def load_scaler(name):
    scaler = joblib.load('../data/{}'.format(name))
    return scaler
