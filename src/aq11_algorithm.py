from inference import infer
import pandas as pd


def compare_sets(pos_row: pd.Series, negatives: pd.DataFrame) -> list[dict]:
    """Identify attributes that differ between positive and negative examples."""
    differences = []
    for _, neg_row in negatives.iterrows():
        diff = compare_two_elements(pos_row, neg_row)
        if diff:
            differences.append(diff)
    return differences

def compare_two_elements(pos_row: pd.Series, neg_row: pd.Series) -> dict[str, ]:
    """Compare two elements and identify differing attributes."""
    return pos_row[pos_row != neg_row].to_dict()

def apply_absorption_law(conditions: list[dict[str, ]]) -> list[dict[str, ]]:
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

def stringify_conditions(conditions: list[dict[str, ]]) -> str:
    """Convert a list of condition dictionaries into a human-readable string format."""
    condition_strings = []
    for condition in conditions:
        condition_str = ' AND '.join([f"'{name}'='{val}'" for name, val in condition.items()])
        condition_strings.append(f"({condition_str})")
    return ' OR '.join(condition_strings)

def generate_rules(df_positive: pd.DataFrame, df_negative: pd.DataFrame) -> list[str]:
    """Generate rules using the AQ11 approach."""
    rules = []
    df_positive_iter = [0]*len(df_positive)

    for i, (_, pos_row) in enumerate(df_positive.iterrows()):
        # Skip positive examples that have already been covered by a rule
        if df_positive_iter[i] == 1:
            continue

        # Compare the positive example with all negative examples
        differences = compare_sets(pos_row, df_negative)
        
        # Apply the absorption law to simplify the rule set
        simplified_conditions = apply_absorption_law(differences)

        # Infer the labels for the positive and negative examples
        preds = infer(stringify_conditions(simplified_conditions), df_positive)
        df_positive_iter = [max(preds[i], df_positive_iter[i]) for i in range(len(preds))]

        # Extend the rules list with the simplified conditions
        rules.extend(simplified_conditions)
    
    final_rules = apply_absorption_law(rules)
    return stringify_conditions(final_rules)

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
