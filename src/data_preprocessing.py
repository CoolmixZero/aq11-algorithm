import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)


def convert_to_categorical(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Converts numerical data into categorical ranges based on quartiles or actual values."""
    data_cp = data.copy()
    unique_values = data_cp[column_name].unique()

    # If there are fewer than 5 unique values, treat them as categories
    if len(unique_values) < 5:
        return data_cp
    
    # Determine the bins for the quartiles, ensuring there are 4 breaks (3 quartile points plus min and max)
    bins = pd.qcut(data_cp[column_name], q=4, duplicates='drop', retbins=True)[1]
    # The labels should be based on the range from each bin to the next
    quartile_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]

    # Convert the numerical column to categorical based on the bins
    data_cp[column_name] = pd.cut(data_cp[column_name], bins=bins, labels=quartile_labels, include_lowest=True)

    return data_cp



def preprocess_data(data: pd.DataFrame, for_training: bool = True) -> dict[str, pd.DataFrame]:
    """Preprocess the dataset."""
    label_column = data.columns[-1]
    processed_data = data.copy()
    for column_name in data.columns[:-1]:
        # Check if the column is numerical
        if processed_data[column_name].dtype.kind in 'ifc':
            processed_data = convert_to_categorical(processed_data, column_name)

    if for_training:
        return {'positive': processed_data[processed_data[label_column] == 1],
                'negative': processed_data[processed_data[label_column] == 0]}
    return processed_data


if __name__ == "__main__":
    data = load_data('../data/student_stress.csv')  # Replace with your file path
    preprocessed_data = preprocess_data(data)
    print(preprocessed_data['positive'])
    print(preprocessed_data['negative'])
