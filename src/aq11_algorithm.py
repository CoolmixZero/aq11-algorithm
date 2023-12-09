from itertools import product
from pprint import pprint


def compare_sets(pos_row, negatives):
    """Identify attributes that differ between positive and negative examples."""
    differences = []
    for _, neg_row in negatives.iterrows():
        diff = compare_two_elements(pos_row, neg_row)
        if diff:
            differences.append(diff)
    return differences

def compare_two_elements(pos_row, neg_row):
    """Compare two elements and identify differing attributes."""
    return pos_row[pos_row != neg_row].to_dict()

def apply_absorption_law(conditions):
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

def create_conditions(df, common_attributes):
    """Create conditions for rules based on common attributes."""
    conditions = []
    for _, row in df.iterrows():
        conditions.append({attr: row[attr] for attr in common_attributes})
    return conditions

def create_rule_from_condition(condition):
    """Generate a rule from a condition."""
    return ' AND '.join([f"{k}='{v}'" for k, v in condition.items()])

def print_conditions(conditions):
    """Print the conditions in a readable format."""
    rules = []
    for condition in conditions:
        rule = ' AND '.join([i for i in condition.values()])
        rules.append(f'({rule})')
    print(' OR '.join(rules))

def generate_rules(df_positive, df_negative):
    """Generate rules using the AQ11 approach."""
    rules = []
    for _, pos_row in df_positive.iterrows():
        differences = compare_sets(pos_row, df_negative)
        print('Before absorption law:')
        print_conditions(differences)
        simplified_conditions = apply_absorption_law(differences)
        print('After absorption law:')
        print_conditions(simplified_conditions)
    return rules