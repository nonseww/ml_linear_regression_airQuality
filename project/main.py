from dataset import load_dataset
from model import run_training

if __name__ == '__main__':
    df = load_dataset('AirQualityUCI.csv')
    learning_rate = 0.001
    epochs = 50
    batch_size = 50
    features = [col for col in df.columns if col not in ['Date', 'Time', 'CO(GT)']]
    label = 'CO(GT)'
    model = run_training(df, features, label, learning_rate, epochs, batch_size)

