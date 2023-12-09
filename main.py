import sys
sys.path.append('src')

from data_preprocessing import load_data, preprocess_data
from aq11_algorithm import aq11_algorithm


def main():
    # Main code to run the AQ11 algorithm
    data = load_data('path_to_dataset.csv')
    preprocessed_data = preprocess_data(data)
    rules = aq11_algorithm(preprocessed_data['positive'], preprocessed_data['negative'])
    

if __name__ == "__main__":
    main()
