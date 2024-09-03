import autopep8
import cn2an
import re
from utils import py_exec_utils


def check_indent_and_run(output_code: str):
    fix_code = autopep8.fix_code(output_code)
    fix_answer = py_exec_utils.execute_with_timeout(fix_code)
    fix_answer_list = fix_answer.split('error:')
    if len(fix_answer_list) == 1:
        return fix_answer_list[0]
    elif fix_answer_list[1].find("unindent") != -1:
        not_fixed = "Indentation exception was not fixed !!!"
        return not_fixed
    else:
        return fix_answer


def get_variables_NLP(output_str: str, question: str):
    split_str = re.split(r"1\. Define variables:|2\. Solution analysis:", output_str)

    if len(split_str) == 3:
        matches = re.findall(r'(.+?):\s*(.+)', split_str[1])
        extracted_variables = {match[0].strip(): match[1].strip() for match in matches}
    else:
        return "invalid format"

    # Listing the extracted variables
    variables_list = "; ".join([f"{var_name} is {var_value}" for var_name, var_value in extracted_variables.items()])

    query_sample = (f"I have extracted the following variables from the question: \nQuestion:{question}\nAnswer:{variables_list}"
                     f"\n\nPlease check if this is correct and complete? If yes, please output True, otherwise output False and point out mistakes.\nAnswer:")
    query_student = (f"I am a student, and I need to extract all the numerical information from a question."
                     f" Here is the question and my answer:\nQuestion:{question}\nAnswer: {variables_list}"
                     f"\n\nPlease check if this is correct and complete? If yes, please output True, otherwise output False and point out mistakes.\nAnswer:")
    query_teacher = (f"As a teacher, I have reviewed my student's work on extracting all the numerical information from a question."
                     f" This is the question and the student's answer: \nQuestion:{question}\nAnswer:{variables_list}"
                     f"\n\nPlease check if this is correct and complete? If yes, please output True, otherwise output False and point out mistakes.\nAnswer:")
    return query_sample


def convert_full_text(query_str: str):
    def convert_percent(match):
        percent_str = match.group(1)
        # Determine the number of decimal places
        decimal_places = len(percent_str.split('.')[1]) if '.' in percent_str else 0
        # Convert to decimal and round to n+2 decimal places
        percentage = round(float(percent_str) / 100, decimal_places + 2)
        return f"{percentage}"

    # 替换百分比
    query_str = re.sub(r'([\d\.]+)%', convert_percent, query_str)

    # 替换阿拉伯数字加中文汉字：如10万
    query_str = re.sub(r'([\d\.]+)(万|千|百|十)',
                       lambda m: str(int(float(m.group(1)) * {'万': 10000, '千': 1000, '百': 100, '十': 10}[m.group(2)])),
                       query_str)

    # 中文数字单位映射
    def convert_cn_numbers_in_string(s):
        cn_num_pattern = re.compile(r"[零一二两三四五六七八九十百千万亿]+")

        # 定义一个函数用于替换匹配的中文数字
        def replace_cn_num(match):
            cn_num = match.group(0)
            if cn_num in ["零", "一"]:
                return match.group()
            if cn_num.startswith('一') and len(cn_num) == 2 and cn_num[1] in '一二三四五六七八九':
                # 保留“一”不变，不转换后面的数字
                try:
                    return '一' + str(cn2an.cn2an(cn_num[1], "normal"))
                except ValueError:
                    # 如果cn2an无法转换，则保留原中文数字
                    return '一' + cn_num[1]
            else:
                return str(cn2an.cn2an(cn_num, "normal"))
        return cn_num_pattern.sub(replace_cn_num, s)

    query_str = convert_cn_numbers_in_string(query_str)

    # 正则表达式，用于匹配分数形式
    def convert_fractions_to_decimals(text):
        fraction_pattern = r'(\d+)/(\d+)'
        fractions = re.findall(fraction_pattern, text)
        for numerator, denominator in fractions:
            decimal = round(int(numerator) / int(denominator), 3)
            text = text.replace(f'{numerator}/{denominator}', str(decimal))
        return text

    query_str = convert_fractions_to_decimals(query_str)

    return query_str


if __name__ == '__main__':
    test_strings = [':"3.15%"', "1/3", "3.15%", "5.8%", "10%", "10万", "三年", "十四", "二十五", "二百", "三千五百", "三千五百万", "五百万美元",
                    "三千五百万美元", "一千万", "有一5年"]
    converted_values = [convert_full_text(s) for s in test_strings]
    print(converted_values)
    read_file = "../dataset/benchmark.txt"
    with open(read_file, 'r', encoding='utf-8') as file_in:
        for query in file_in:
            query = convert_full_text(query)
            print(query)
