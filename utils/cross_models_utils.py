import re


def check_json_completeness(json_data):
    required_keys = ["Target Variable(s)", "Input Variable(s)", "List all formula(s)/equation(s)"]
    for key in required_keys:
        if key not in json_data:
            return False
    return True


def print_paths(paths):
    output = ""
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, path in enumerate(paths):
        if i < len(letters):
            output += f"\nSolution {letters[i]}:"
            for value in path:
                output += f"\t{value}\n"
        else:
            pass
    return output


def letter_paths(paths):
    output_dict = {}
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, path in enumerate(paths):
        if i < len(letters):
            path_description = f"Solution {letters[i]}:"
            for value in path:
                path_description += f"\t{value}\n"
            output_dict[f"Solution {letters[i]}"] = path_description
        else:
            pass
    return output_dict


def remove_sympy_error_none(result_list, solution_paths, all_formula_paths):
    for i in range(len(result_list) - 1, -1, -1):
        result = result_list[i]
        if result is None or str(result).startswith("Error"):
            del result_list[i]
            del solution_paths[i]
            del all_formula_paths[i]


def remove_sympy_not_num(result_list, solution_paths, all_formula_paths):
    for i in range(len(result_list) - 1, -1, -1):
        result = result_list[i]
        if not str_is_number(result):
            del result_list[i]
            del solution_paths[i]
            del all_formula_paths[i]
        else:
            result_list[i] = float(result)


def remove_sympy_0(result_list, solution_paths, all_formula_paths):
    for i in range(len(result_list) - 1, -1, -1):
        if result_list[i] == 0:
            del result_list[i]
            del solution_paths[i]
            del all_formula_paths[i]


def remove_sympy_negative(result_list, solution_paths, all_formula_paths):
    for i in range(len(result_list) - 1, -1, -1):
        if result_list[i] < 0:
            del result_list[i]
            del solution_paths[i]
            del all_formula_paths[i]


def str_is_number(s):
    try:
        float(s)
        return True
    except Exception:
        return False


def replace_multiple_spaces_with_underscores(expression):
    pattern = r'\b[A-Za-z]+(?:\s+[A-Za-z]+)+\b'
    replace_func = lambda match: '_'.join(match.group().split())
    new_expression = re.sub(pattern, replace_func, expression)
    return new_expression


def check_and_correct_parentheses(expression):
    parts = expression.split('=')
    if len(parts) != 2:
        return expression
    parts[1] = parts[1].strip()
    left_paren_count = parts[1].count('(')
    right_paren_count = parts[1].count(')')

    if left_paren_count > right_paren_count:
        parts[1] = parts[1] + ')' * (left_paren_count - right_paren_count)
    else:
        parts[1] = '(' * (right_paren_count - left_paren_count) + parts[1]

    return parts[0] + '= ' + parts[1]
