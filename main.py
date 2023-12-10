import sys
sys.path.append('src')
from data_preprocessing import load_data, preprocess_data
from aq11_algorithm import generate_rules
from inference import evaluate_performance


def main():
    # Main code to run the AQ11 algorithm
    eval_samples = 100
    data = load_data('data/student_stress.csv').iloc[eval_samples:]
    preprocessed_data = preprocess_data(data)
    print(preprocessed_data['positive'].head())
    print(preprocessed_data['negative'].head())

    print("Training AQ11 algorithm...")
    rules = generate_rules(preprocessed_data['positive'].iloc[:, :-1], preprocessed_data['negative'].iloc[:, :-1])
    with open('data/rules.txt', 'w') as f:
        f.write(rules)
    print("Generated Rules!")
    print("Evaluating performance...")
    metrics = evaluate_performance('data/rules.txt', 'data/student_stress.csv', 'stress_level', eval_samples)
    print(metrics)

if __name__ == "__main__":
    main()
