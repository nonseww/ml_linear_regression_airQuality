from dataset import load_dataset
from model import run_training
from save_load import save_model, load_model
from prediction import predict_CO, show_predictions

if __name__ == '__main__':
    df = load_dataset('AirQualityUCI.csv')
    need_train = False
    # need_train = True
    if need_train:
        # one-feature
        learning_rate_1 = 0.001
        epochs_1 = 50
        batch_size_1 = 50
        features_1 = [col for col in df.columns if col not in ['Date', 'Time', 'CO(GT)', 'PT08.S3(NOx)', 'NO2(GT)']]
        label_1 = 'CO(GT)'
        model_1 = run_training(df, features_1, label_1, learning_rate_1, epochs_1, batch_size_1)

        save_model(model_1, "linear_regression_airQuality_1.keras")

        # two-features
        learning_rate_2 = 0.001
        epochs_2 = 50
        batch_size_2 = 50
        features_2 = [col for col in df.columns if col not in ['Date', 'Time', 'CO(GT)', 'PT08.S3(NOx)']]
        label_2 = 'CO(GT)'
        model_2 = run_training(df, features_2, label_2, learning_rate_2, epochs_2, batch_size_2)

        save_model(model_2, "linear_regression_airQuality_2.keras")

        # three-features
        learning_rate_3 = 0.001
        epochs_3 = 50
        batch_size_3 = 50
        features_3 = [col for col in df.columns if col not in ['Date', 'Time', 'CO(GT)']]
        label_3 = 'CO(GT)'
        model_3 = run_training(df, features_3, label_3, learning_rate_3, epochs_3, batch_size_3)

        save_model(model_3, "linear_regression_airQuality_3.keras")
    else:
        model = load_model("linear_regression_airQuality_3.keras")
        features = [col for col in df.columns if col not in ['Date', 'Time', 'CO(GT)']]
        label = "CO(GT)"
        output = predict_CO(model, df, features, label)
        show_predictions(output)
