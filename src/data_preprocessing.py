import pandas as pd
from sklearn.model_selection import train_test_split


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
    
    X_train, X_test, y_train, y_test = \
        train_test_split(data.iloc[:, :-1], data[label_column], test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data = load_data('../data/example.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)