from dataset import load_dataset, generate_correlation_matrix

if __name__ == '__main__':
    df = load_dataset('AirQualityUCI.csv')
    generate_correlation_matrix(df)

