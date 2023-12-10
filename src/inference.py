import pandas as pd


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

def apply_rule_to_sample(rule: str, sample: pd.Series) -> bool:
    """
    Evaluates if the sample satisfies the given rule.
    """
    or_split = [rule] if ' OR ' not in rule else rule.split(' OR ')

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
    for _, sample in new_samples.iterrows():
        predictions.append(apply_rule_to_sample(rules, sample))
    return predictions