import sys
sys.path.append('src')
from data_preprocessing import load_data, preprocess_data
from aq11_algorithm import generate_rules, evaluate_instances


def main():
    # Main code to run the AQ11 algorithm
    data = load_data('data/test_rules.csv')
    preprocessed_data = preprocess_data(data)
    print(preprocessed_data['positive'])
    print(preprocessed_data['negative'])

    print("Training AQ11 algorithm...")
    rules = generate_rules(preprocessed_data['positive'].iloc[:, :-1], preprocessed_data['negative'].iloc[:, :-1])
    print("Generated Rules:", rules)

    # print("Evaluation on new instances...")
    # Convert new instances to DataFrame and preprocess
    # new_data = load_data('data/student_stress.csv')
    # preprocessed_new_data = preprocess_data(new_data)

    # # Convert DataFrames to lists of dictionaries for evaluation
    # positive_instances = preprocessed_new_data['positive'].iloc[:, :-1].to_dict(orient='records')
    # negative_instances = preprocessed_new_data['negative'].iloc[:, :-1].to_dict(orient='records')

    # # Evaluate instances
    # positive_predictions = evaluate_instances(rules, positive_instances)
    # negative_predictions = evaluate_instances(rules, negative_instances)

    # print("Positive Predictions:", positive_predictions)
    # print("Negative Predictions:", negative_predictions)



if __name__ == "__main__":
    main()
