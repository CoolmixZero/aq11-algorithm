def apply_rule_to_sample(rule, sample):
    """
    Evaluates a single rule against a sample.
    
    Parameters:
    rule (str): The rule represented as a string of conditions.
    sample (pd.Series): The data sample to evaluate the rule against.

    Returns:
    bool: True if the rule applies to the sample, False otherwise.
    """
    # Split the rule by 'AND' to evaluate each condition
    conditions = rule.split(' AND ')
    for condition in conditions:
        # Each condition is expected to be in the form 'attribute=value'
        attribute, value = condition.split('=')
        value = value.strip("'")  # Remove quotes around the value if present
        # Check if the sample does not satisfy the condition
        if str(sample[attribute]) != value:
            return False
    return True

def infer(rules, new_samples):
    """
    Applies a set of rules to new data samples to infer outcomes.

    Parameters:
    rules (list): A list of rules represented as strings.
    new_samples (pd.DataFrame): New data samples to apply the rules to.

    Returns:
    list: Inferred outcomes for each sample.
    """
    predictions = []
    for _, sample in new_samples.iterrows():
        # Initialize the prediction as negative (assuming binary classification)
        prediction = '-'
        for rule in rules:
            if apply_rule_to_sample(rule, sample):
                # If any rule applies, the prediction is positive
                prediction = '+'
                break  # No need to check other rules if one matches
        predictions.append(prediction)
    return predictions

# Example usage:
# Assuming `rules` is a list of strings representing the rules,
# and `new_data` is a pandas DataFrame containing new samples:
# predictions = infer(rules, new_data)