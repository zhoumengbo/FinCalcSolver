from utils.tree_utils import print_tree, DFSBuilder
import re

from utils.cross_models_utils import str_is_number
from utils.sympy_for_paths import sympy_reserved


def create_tree_and_paths(logger, q_json, formula_list):
    input_v_list = []
    for input_v in q_json["Input Variable(s)"]:
        input_v_list.append(input_v["Variable Name"].replace("'", ""))

    target_builder = []
    no_visited_target = []
    for target_v in q_json["Target Variable(s)"]:
        no_visited_target.append(target_v["Variable Name"].replace("'", ""))

    while no_visited_target:
        target_v_name = no_visited_target.pop(0)
        created = False
        builder = DFSBuilder(target_v_name)

        while True:
            # node is variable
            if builder.get_current_n_type() == "variable":
                variable = builder.get_current_value()
                while variable in input_v_list or variable.isdigit():
                    if not builder.move_to_first_unvisited_child():
                        created = True
                        break
                    else:
                        variable = builder.get_current_value()
                if created:
                    created = False
                elif builder.get_current_n_type() == "variable":
                    node_num = 0
                    for formula in formula_list:
                        variables_list = extract_variables(formula)
                        if variable in variables_list and not check_common_elements(variables_list, no_visited_target):
                            if "=" in formula.strip():
                                builder.add_child(formula.strip(), "formula")
                                node_num += 1
                    if node_num == 0:
                        for formula in formula_list:
                            variables_list = extract_variables(formula)
                            if variable in variables_list:
                                if "=" in formula.strip():
                                    builder.add_child(formula.strip(), "formula")
            # node is formula
            else:
                current_formula = builder.get_current_value()
                values_list = extract_variables(current_formula)
                if len(values_list) == 0:
                    logger.info("Can not extract variables from formula: " + current_formula)
                else:
                    for value in values_list:
                        if value.strip() not in sympy_reserved:
                            builder.add_child(value, "variable")
            if not builder.continue_dfs():
                logger.info("### Tree ###\n{0}\nCurrent_key: ({1})".format(print_tree(builder.root),
                                                                           builder.get_current_key()))
                break
            if builder.get_current_key() > 2000:
                logger.info("### Tree ###\n{0}\nCurrent_key: ({1})".format(print_tree(builder.root),
                                                                           builder.get_current_key()))
                builder.key_index = 0
                break
        target_builder.append(builder)
        input_v_list.append(target_v_name)
    return target_builder


def extract_formula_paths(node, current_path=None, all_paths=None):
    if current_path is None:
        current_path = []
    if all_paths is None:
        all_paths = []
    new_path = current_path.copy()
    if node.n_type == "formula":
        new_path.append(node.value)
    if node.children:
        if node.n_type == "formula":
            child_paths_lists = [extract_formula_paths(child, None, None) for child in node.children]
            combined_paths = combine_paths(child_paths_lists)
            for combined_path in combined_paths:
                all_paths.append(new_path + combined_path)
        else:
            for child in node.children:
                extract_formula_paths(child, new_path, all_paths)
    else:
        all_paths.append(new_path)
    return all_paths


def extract_paths(node, value_dict, current_path=None, all_paths=None):
    if current_path is None:
        current_path = []
    if all_paths is None:
        all_paths = []

    new_path = current_path.copy()
    if node.n_type == "variable":
        if not node.children:
            if not node.value.isdigit():
                value = value_dict.get(node.value, None)
                new_path.append("\t{0}: {1}".format(node.value, value))
        else:
            new_path.append("")
            new_path.append("Calculate {0}".format(node.value))
    if node.n_type == "formula":
        new_path.append(("Formula/equation: " + node.value))

    if node.children:
        if node.n_type == "formula":
            child_paths_lists = [extract_paths(child, value_dict, [], None) for child in node.children]
            combined_paths = combine_paths(child_paths_lists)
            for combined_path in combined_paths:
                all_paths.append(new_path + combined_path)
        else:
            for child in node.children:
                extract_paths(child, value_dict, new_path, all_paths)
    else:
        all_paths.append(new_path)
    return all_paths


def combine_paths(child_paths_lists):
    if not child_paths_lists:
        return [[]]

    combined = []
    first = child_paths_lists[0]
    rest_combinations = combine_paths(child_paths_lists[1:])
    for path in first:
        for rest in rest_combinations:
            combined.append(path + rest)
    return combined


def extract_variables(expression):
    parts = expression.split('=')
    if len(parts) != 2:
        return []
    tokens = re.split(r'[=+\-*/^,()]', expression)
    variables = []
    for token in tokens:
        if token.strip():
            if not str_is_number(token) and token not in sympy_reserved:
                variables.append(token.strip())
    return variables


def check_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return bool(set1.intersection(set2))
