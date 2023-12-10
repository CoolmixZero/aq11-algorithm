import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def convert_to_categorical(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Converts numerical data into categorical ranges based on quartiles."""
    # Determine the quartiles
    quartiles = pd.qcut(data[column_name], q=4, duplicates='drop').unique()
    quartile_labels = [f"Q{i+1}" for i in range(len(quartiles))]
    
    # Convert the numerical column to categorical
    data[column_name] = pd.qcut(data[column_name], q=4, labels=quartile_labels, duplicates='drop')
    return data

def preprocess_data(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Preprocess the dataset."""
    label_column = data.columns[-1]
    for column_name in data.columns[:-1]:
        # Check if the column is numerical
        if data[column_name].dtype.kind in 'ifc':
            data = convert_to_categorical(data, column_name)
    
    return {'positive': data[data[label_column] == 1], 'negative': data[data[label_column] == 0]}

if __name__ == "__main__":
    data = load_data('../data/example.csv')
    preprocessed_data = preprocess_data(data)
    print(preprocessed_data['positive'])
    print(preprocessed_data['negative'])