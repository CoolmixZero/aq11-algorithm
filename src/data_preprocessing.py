import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)


def convert_to_categorical(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Converts numerical data into categorical ranges based on quartiles."""
    # Determine the quartiles
    quartiles, bins = pd.qcut(data[column_name], q=4, duplicates='drop', retbins=True)
    quartile_labels = [f"Q{i + 1}" for i in range(len(bins) - 1)]

    # Convert the numerical column to categorical
    data = data.copy()
    data[column_name] = pd.qcut(data[column_name], q=4, labels=quartile_labels, duplicates='drop')
    return data


def preprocess_data(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Preprocess the dataset."""
    label_column = data.columns[-1]
    processed_data = data.copy()
    for column_name in data.columns[:-1]:
        # Check if the column is numerical
        if processed_data[column_name].dtype.kind in 'ifc':
            processed_data = convert_to_categorical(processed_data, column_name)

    return {'positive': processed_data[processed_data[label_column] == 1],
            'negative': processed_data[processed_data[label_column] == 0]}


if __name__ == "__main__":
    data = load_data('../data/student_stress.csv')  # Replace with your file path
    preprocessed_data = preprocess_data(data)
    print(preprocessed_data['positive'])
    print(preprocessed_data['negative'])
