import pandas as pd


def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def convert_to_categorical(data, column):
    """Converts numerical data into categorical ranges based on quartiles."""
    # Determine the quartiles
    quartiles = pd.qcut(data[column], q=4, duplicates='drop').unique()
    quartile_labels = [f"Q{i+1}" for i in range(len(quartiles))]
    
    # Convert the numerical column to categorical
    data[column] = pd.qcut(data[column], q=4, labels=quartile_labels, duplicates='drop')
    return data

def preprocess_data(data):
    """Preprocess the dataset."""
    label_column = data.columns[-1]
    for column in data.columns[:-1]:
        # Check if the column is numerical
        if data[column].dtype.kind in 'ifc':
            data = convert_to_categorical(data, column)
    
    return {'positive': data[data[label_column] == 1], 'negative': data[data[label_column] == 0]}

if __name__ == "__main__":
    data = load_data('../data/example.csv')
    preprocessed_data = preprocess_data(data)
    print(preprocessed_data['positive'])
    print(preprocessed_data['negative'])