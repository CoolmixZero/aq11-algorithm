import sys
sys.path.append('src')
from data_preprocessing import load_data, preprocess_data
from aq11_algorithm import generate_rules


def main():
    # Main code to run the AQ11 algorithm
    data = load_data('data/student_stress.csv')
    preprocessed_data = preprocess_data(data)
    print(preprocessed_data['positive'].head())
    print(preprocessed_data['negative'].head())

    print("Training AQ11 algorithm...")
    rules = generate_rules(preprocessed_data['positive'].iloc[:, :-1], preprocessed_data['negative'].iloc[:, :-1])
    with open('data/rules.txt', 'w') as f:
        f.write(rules)
    print("Generated Rules!")


if __name__ == "__main__":
    main()
