import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_preprocessing import load_data, preprocess_data


def parse_condition(condition: str):
    """
    Parses a single condition and returns the attribute and value.
    Adds error checking to ensure the condition is well-formed.
    """
    parts = condition.split('=')
    if len(parts) != 2:
        raise ValueError(f"Condition '{condition}' is not well-formed.")
    attribute, value = parts
    attribute = attribute.strip("'")
    value = value.strip("'")
    return attribute, value

def apply_condition_to_sample(attribute: str, value: str, sample: pd.Series) -> bool:
    """
    Applies a single condition to a sample.
    """
    return str(sample[attribute]) == value

def apply_rule_to_sample(rules: str, sample: pd.Series) -> bool:
    """
    Evaluates if the sample satisfies the given rules.
    """
    or_split = [rules] if ' OR ' not in rules else rules.split(' OR ')

    for disjunct in or_split:
        conjunct_conditions = disjunct.strip("()")
        conjunct_conditions = [conjunct_conditions] if ' AND ' not in conjunct_conditions \
                              else conjunct_conditions.split(' AND ')
        if all(apply_condition_to_sample(*parse_condition(cond), sample) for cond in conjunct_conditions):
            return 1
    return 0

def infer(rules: str, new_samples: pd.DataFrame) -> list:
    """
    Apply stringified rules to new samples to infer the outcomes.
    """
    predictions = []
    samples_iter = new_samples.iterrows()
    if len(rules.split(' OR ')) > 1000:
        samples_iter = tqdm(samples_iter, total=len(new_samples))
        samples_iter.set_description("Inferring")

    for _, sample in samples_iter:
        predictions.append(apply_rule_to_sample(rules, sample))
    return predictions

def evaluate_performance(rules: str, data_path: str, true_labels_column: str, 
                         samples_num: int = 100) -> dict:
    """
    Evaluates the performance of the rules.

    Parameters:
    rules (str or list): The rules as a string or a list of strings.
    data_path (str): The path to the dataset (CSV file).
    true_labels_column (str): The name of the column containing the true labels.

    Returns:
    dict: A dictionary containing various performance metrics.
    """

    # Load rules
    if isinstance(rules, str):
        with open(rules, 'r') as file:
            rules = file.read()

    # Load the data
    data = load_data(data_path)
    data = preprocess_data(data, for_training=False).iloc[:samples_num]
    true_labels = data[true_labels_column]
    data = data.drop(columns=[true_labels_column])

    # Apply inference to get predictions
    predictions = infer(rules, data)

    # Calculate performance metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, pos_label=1)
    recall = recall_score(true_labels, predictions, pos_label=1)
    f1 = f1_score(true_labels, predictions, pos_label=1)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len([1, 0])))
    ax.set_yticks(range(len([1, 0])))
    ax.set_xticklabels([1, 0])
    ax.set_yticklabels([1, 0])
    for i in range(len([1, 0])):
        for j in range(len([1, 0])):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()

    # Return the metrics
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}