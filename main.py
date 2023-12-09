import sys
sys.path.append('src')

from data_preprocessing import load_data, preprocess_data
from aq11_algorithm import generate_rules


def main():
    # Main code to run the AQ11 algorithm
    data = load_data('data/example.csv')
    preprocessed_data = preprocess_data(data)
    print(preprocessed_data['positive'])
    print(preprocessed_data['negative'])
    generate_rules(preprocessed_data['positive'].iloc[:, :-1], preprocessed_data['negative'].iloc[:, :-1])

if __name__ == "__main__":
    main()
