import json
from decimal import Decimal, ROUND_HALF_UP
from sympy import symbols, sympify
import re
import ast
import operator as op

replacements = {
    'I': 'I_var',
    'E': 'E_var',
    'S': 'S_var'
}
content_list_6 = [
        "1. Define Input Variable(s)",
        "2. Input Variable(s) Assignment",
        "3. Define Intermediate Variable(s)",
        "4. Define Target Variable(s)",
        "5. Intermediate Variable Calculate Formula(s) Step By Step",
        "6. Target Variable Calculate Formula(s)"
    ]

content_list_5 = [
        "1. Define Input Variable(s)",
        "2. Input Variable(s) Assignment",
        "3. Define Intermediate Variable(s)",
        "4. Define Target Variable(s)",
        "5. List the equation(s) that include all defined variables"
    ]


def json_6_to_sympy(input_string):
    data_dict = fix_str_to_json(input_string)
    variables = data_dict[content_list_6[0]]
    values_dict = data_dict[content_list_6[1]]
    target_variables = data_dict[content_list_6[3]]
    intermediate_formulas = {}
    target_formulas = {}
    if content_list_6[4] in data_dict:
        intermediate_formulas = data_dict[content_list_6[4]]
    if content_list_6[5] in data_dict:
        target_formulas = data_dict[content_list_6[5]]
    variables_values_set = set(variables.values())
    values_dict_keys_set = set(values_dict.keys())
    is_consistent = variables_values_set == values_dict_keys_set
    if is_consistent is False:
        return "The variable definitions in the first two steps are inconsistent !!!"

    sympy_input_variables = symbols(variables_values_set)
    for formula in intermediate_formulas.values():
        formula = formula.strip()
        left, right = formula.split('=')
        variables_defined = check_variables_in_lists(right, values_dict.keys())
        if variables_defined is not True:
            return "Variable {0} in formula are undefined !!!".format(variables_defined)
        formula = sympify(right.replace('^', '**'))
        computed_value = formula.subs(values_dict).evalf()
        values_dict[left.strip()] = format_num(computed_value)

    for formula in target_formulas.values():
        formula = formula.strip()
        left, right = formula.split('=')
        variables_defined = check_variables_in_lists(right, values_dict.keys())
        if variables_defined is not True:
            return "Variable {0} in formula are undefined !!!".format(variables_defined)
        formula = sympify(right.replace('^', '**'))
        computed_value = formula.subs(values_dict)
        values_dict[left.strip()] = format_num(computed_value)

    swapped_target_variables = {value: key for key, value in target_variables.items()}
    output_str = ''
    for key in target_variables.values():
        if key in values_dict:
            name = swapped_target_variables.get(key, "Unknown")
            output_str += f"{name} is defined as variable {key}, and its calculation result is {values_dict[key]}.\n"
        else:
            return f"Key '{key}' not found in the dictionary."
    return output_str


def fix_str_to_json(input_string):
    input_string_processed = process_assignment_section(input_string)
    try:
        json_data = json.loads(input_string_processed)
        json_data = replace_with_var(json_data)
    except json.JSONDecodeError as e:
        json_data = f"Error converting to JSON: {str(e)}"
    return json_data


def check_variables_in_lists(formula, variables):
    extracted_vars = re.findall(r'\b[A-Za-z][A-Za-z0-9]*\b', formula)
    for var in extracted_vars:
        if var not in variables:
            return var
    return True


def format_num(computed_value):
    value_as_decimal = Decimal(float(computed_value))
    two_places = Decimal('0.01')
    value_rounded = value_as_decimal.quantize(two_places, rounding=ROUND_HALF_UP)
    formatted_num = format(value_rounded, 'f').rstrip('0').rstrip('.')
    return formatted_num


def replace_with_var(obj):
    obj = replace_with_2key(obj)
    obj = replace_with_value(obj)
    return obj


def replace_with_2key(obj):
    if "2. Input Variable(s) Assignment" in obj:
        assignment_data = obj["2. Input Variable(s) Assignment"]
        for key, value in list(assignment_data.items()):
            new_key = replace_multiple(key)
            if new_key != key:
                del assignment_data[key]
            assignment_data[new_key] = value
    return obj


def replace_with_value(obj):
    if isinstance(obj, dict):
        return {k: replace_with_value(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_with_value(element) for element in obj]
    elif isinstance(obj, str):
        return replace_multiple(obj)
    else:
        return obj


def replace_multiple(obj):
    for old, new in replacements.items():
        obj = obj.replace(old, new)
    return obj


def safe_eval(expr):
    operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                 ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
                 ast.USub: op.neg}

    def eval_(node):
        if isinstance(node, ast.Num):  # <number>
            return node.n
        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
            return operators[type(node.op)](eval_(node.left), eval_(node.right))
        else:
            raise TypeError(node)

    return eval_(ast.parse(expr, mode='eval').body)


def evaluate_expression(expr, context):
    """
    Safely evaluate an arithmetic expression based on the provided context.

    :param expr: String, the arithmetic expression to be evaluated.
    :param context: Dictionary, a mapping of variables to their values.
    :return: Evaluated result of the expression.
    """
    try:
        # Replace variables in the expression with their values from the context
        for var in context:
            expr = expr.replace(var, str(context[var]))

        # Safely evaluate the expression
        return format_num(safe_eval(expr))
    except Exception as e:
        return None


def process_assignment_section(input_str):
    """
    Process the '2. Input Variable(s) Assignment' section of the input string.

    :param input_str: String containing the '2. Input Variable(s) Assignment' section.
    :return: Processed string with evaluated expressions.
    """
    # Extract the assignment section
    assignment_section_match = re.search(r'"2\. Input Variable\(s\) Assignment": \{(.*?)\}', input_str, re.DOTALL)
    assignment_section = assignment_section_match.group(1) if assignment_section_match else ""

    # Parse the assignments into a dictionary
    assignment_dict = {}
    for line in assignment_section.splitlines():
        match = re.match(r'\s*"(\w+)": (.*?)(,)?$', line)
        if match:
            var, value = match.group(1), match.group(2)
            assignment_dict[var] = value

    # Evaluate expressions in the assignment_dict
    for var, expr in assignment_dict.items():
        # Check if the expression is a calculation
        if "*" in expr or "/" in expr or "+" in expr or "-" in expr:
            result = evaluate_expression(expr, assignment_dict)
            if result is not None:
                assignment_dict[var] = result

    # Reconstruct the assignment section with evaluated values
    updated_assignment_section = ',\n'.join([f'    "{var}": {assignment_dict[var]}' for var in assignment_dict])
    updated_assignment_section = '"2. Input Variable(s) Assignment": {\n' + updated_assignment_section + '\n  }'

    # Replace the original assignment section in the input string
    updated_input_str = re.sub(r'"2\. Input Variable\(s\) Assignment": \{.*?\}', updated_assignment_section, input_str,
                               flags=re.DOTALL)

    return updated_input_str
