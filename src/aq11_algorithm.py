from itertools import product
from pprint import pprint

import pandas as pd


def compare_sets(pos_row: pd.Series, negatives: pd.DataFrame) -> list[dict]:
    """Identify attributes that differ between positive and negative examples."""
    differences = []
    for _, neg_row in negatives.iterrows():
        diff = compare_two_elements(pos_row, neg_row)
        if diff:
            differences.append(diff)
    return differences

def compare_two_elements(pos_row: pd.Series, neg_row: pd.Series) -> dict[str, str]:
    """Compare two elements and identify differing attributes."""
    return pos_row[pos_row != neg_row].to_dict()

def apply_absorption_law(conditions: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Applies the absorption law to a list of conditions to simplify the rule set.
    If one condition is a subset of another, it absorbs the other.
    """
    # Start with all conditions marked as not absorbed
    absorbed = [False] * len(conditions)
    
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if i != j and not absorbed[i] and not absorbed[j]:
                # Check if cond1 is a subset of cond2 (cond1 absorbs cond2)
                if all(item in cond2.items() for item in cond1.items()):
                    absorbed[j] = True
                # Check if cond2 is a subset of cond1 (cond2 absorbs cond1)
                elif all(item in cond1.items() for item in cond2.items()):
                    absorbed[i] = True
    
    # Filter out absorbed conditions
    simplified_conditions = [cond for i, cond in enumerate(conditions) if not absorbed[i]]
    
    return simplified_conditions

def print_conditions(conditions: list[dict[str, str]]) -> None:
    """Print the conditions in a readable format."""
    rules = []
    for condition in conditions:
        rule = ' AND '.join([f'{name}={val}' for name, val in condition.items()])
        rules.append(f'({rule})')
    print(' OR '.join(rules))

def generate_rules(df_positive: pd.DataFrame, df_negative: pd.DataFrame) -> list[str]:
    """Generate rules using the AQ11 approach."""
    rules = []
    for _, pos_row in df_positive.iterrows():
        # Compare the positive example with all negative examples
        differences = compare_sets(pos_row, df_negative)
        
        print('Before absorption law:')
        print_conditions(differences)
        
        # Apply the absorption law to simplify the rule set
        simplified_conditions = apply_absorption_law(differences)
        
        print('After absorption law:')
        print_conditions(simplified_conditions)
        
        # Extend the rules list with the simplified conditions
        rules.extend(simplified_conditions)
        
    return rules

def evaluate_instances(rules: list[dict[str, str]], instances: list[dict[str, str]]) -> list[int]:
    """
    Evaluate new instances using the generated rules.
    Return a list of predicted labels (0 or 1) for each instance.
    """
    predictions = []

    for instance in instances:
        # Check if the instance satisfies any of the rules
        for rule in rules:
            if all(item in instance.items() for item in rule.items()):
                # If the instance satisfies a rule, predict the corresponding label
                predictions.append(1)
                break
        else:
            # If the instance does not satisfy any rule, predict label 0
            predictions.append(0)

    return predictions
