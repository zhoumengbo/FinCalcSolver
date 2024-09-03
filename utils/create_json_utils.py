import json
import os
import re


def save_json(data, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=2)
        return True
    except Exception as e:
        return e


def format_variable_name(json_str):
    try:
        data = json.loads(json_str.strip())
    except Exception:
        try:
            json.loads(json_str.strip().replace("None", "null"))
            return format_variable_name(json_str.strip().replace("None", "null"))
        except Exception as e:
            print(f"Exception in json.loads: {e}")
            return json_str
    replacements_done = False

    for key in ["Input Variable(s)", "Target Variable(s)"]:
        if key in data:
            for variable in data[key]:
                v_n = variable["Variable Name"].strip()
                if " " in v_n or "-" in v_n:
                    variable["Variable Name"] = v_n.replace(" ", "_").replace("-", "_")
                    replacements_done = True
    if replacements_done:
        return json.dumps(data)
    else:
        return json_str


def extract_json_from_str(json_str):
    json_pattern = r'{.*}'
    match = re.search(json_pattern, json_str, re.DOTALL)
    data = None
    try:
        data = json.loads(match.group().strip())
    except Exception as e:
        print(f"Exception in json.loads: {e}")
    return data


def extract_variable_name(*json_strings):
    variable_name_dict = {}
    target_v = []
    input_v = []
    for json_str in json_strings:
        json_pattern = r'{.*}'
        match = re.search(json_pattern, json_str, re.DOTALL)
        try:
            data = json.loads(match.group().strip())
            for key in ["Input Variable(s)", "Target Variable(s)", "Intermediate Variable(s)"]:
                if key in data:
                    for variable in data[key]:
                        v_n = variable["Variable Name"].strip()
                        variable_name_dict[v_n] = variable["General Financial Interpretation for Variable"]
                        if key == "Target Variable(s)":
                            target_v.append(v_n)
                        elif key == "Input Variable(s)":
                            input_v.append(v_n)
        except Exception as e:
            print(f"Exception in json.loads: {e}")
    return variable_name_dict, target_v, input_v


def format_formulas(json_str):
    try:
        data = json.loads(json_str.strip())
    except Exception as e:
        return "Error in json.loads: {0}".format(str(e))

    try:
        formulas = data["List all formula(s)/equation(s)"]
        formula_list = []
        for key, value in formulas.items():
            parts = value.split('=')
            if len(parts) > 2:
                value = f"{parts[0]} = {parts[1]}"
            elif len(parts) == 1:
                if key.startswith("eq") or key == "None":
                    continue
                else:
                    value = f"{key} = {value}"
            formula_list.append(value)
    except Exception:
        formulas = extract_root_elements(data)
        formula_list = []
        for formula in formulas:
            parts = formula.split('=')
            if len(parts) == 1:
                continue
            if len(parts) > 2:
                formula = f"{parts[0]} = {parts[1]}"
            formula_list.append(formula)
    return formula_list


def extract_root_elements(data):
    elements = []
    if isinstance(data, dict):
        for value in data.values():
            elements.extend(extract_root_elements(value))
    elif isinstance(data, list):
        for item in data:
            elements.extend(extract_root_elements(item))
    else:
        elements.append(data)
    return elements


def merge_json_strings(*json_strings):
    merged_dict = {}
    for json_str in json_strings:
        if isinstance(json_str, dict):
            merged_dict.update(json_str)
        else:
            json_pattern = r'{.*}'
            match = re.search(json_pattern, json_str, re.DOTALL)
            try:
                data = json.loads(match.group().strip())
                merged_dict.update(data)
            except Exception as e:
                print(f"Exception in json.loads: {e}")
    return merged_dict


def print_history(data):
    string_representation = ["\n\n### {}\n### Input:\n{}\n\n### Output:\n{}\n".format(name, inp, out) for name, inp, out
                             in data]
    return "\n".join(string_representation)