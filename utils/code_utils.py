import math
import re


def transform_code_with_results(original_code):
    # 正则表达式模式，用于查找后面跟有数学表达式的变量赋值。
    assignment_pattern = re.compile(r'(\w+(?:\s*,\s*\w+)*)\s*=\s*([^#\n]+)')
    # 将代码分割成行进行处理。
    lines = original_code.split('\n')
    # 用于跟踪变量值的字典。
    variables = {}
    # 遍历行并处理它们。
    transformed_lines = []
    for line in lines:
        # 检查行是否包含我们想要计算的赋值。
        match = assignment_pattern.search(line)
        if match:
            all_keys, value = match.groups()
            # 删除行内注释并去除表达式两端的空格。
            value = value.split('#')[0].strip()
            # 拆分多个变量。
            keys = [key.strip() for key in all_keys.split(',')]
            # 尝试计算表达式。
            try:
                # 安全地评估表达式。
                value_evaluated = eval(value, {"math": math, "e": math.exp, "__builtins__": __builtins__}, variables)
                # 处理多变量赋值的情况。
                if len(keys) > 1 and isinstance(value_evaluated, tuple):
                    # 计算原始行的缩进
                    indentation = len(line) - len(line.lstrip())
                    indent_str = ' ' * indentation
                    transformed_line = f'{line}\n'
                    for i, key in enumerate(keys):
                        variables[key] = value_evaluated[i]
                        transformed_line += f'{indent_str}{key} = {value_evaluated[i]}  # AutoGen\n'
                else:
                    # 将计算后的值存储在字典中。
                    variables[keys[0]] = value_evaluated
                    # 如果是计算表达式，则添加原始行和计算结果。
                    if any(i in value for i in ['+', '-', '*', '/', 'math']):
                        # 将数学表达式中的变量名替换为其值。
                        sorted_variables = sorted(variables.items(), key=lambda x: len(x[0]), reverse=True)
                        for var_name, var_value in sorted_variables:
                            if isinstance(var_value, (int, float)):
                                # value = value.replace(var_name, str(var_value))
                                pattern = r'\b(?<!\bmath\.)' + re.escape(var_name) + r'\b'
                                value = re.sub(pattern, str(var_value), value)
                        # 计算原始行的缩进
                        indentation = len(line) - len(line.lstrip())
                        indent_str = ' ' * indentation
                        # 使用新的缩进来构造transformed_line
                        transformed_line = f'{line}\n{indent_str}{keys[0]} = {value}  #AutoGen#\n{indent_str}{keys[0]} = {value_evaluated}  #AutoGen#'
                    else:
                        transformed_line = line
            except:
                # 如果不是数学表达式，只存储变量本身。
                variables[keys[0]] = value
                transformed_line = line
        else:
            transformed_line = line
        transformed_lines.append(transformed_line)

    # 将转换后的行连接回单个字符串。
    return '\n'.join(transformed_lines)
