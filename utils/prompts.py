en_prompts_test_convert_question_1 = {
    'CoT': open("prompts/convert_question/CoT", 'r').read(),
    'PoT': open("prompts/convert_question/PoT", 'r').read(),
}

en_prompts_test_convert_question_2 = {
    'CoT': open("prompts/convert_question/CoT_2", 'r').read(),
    'CoT_Q': open("prompts/convert_question/CoT_2_Q", 'r').read(),
    'PoT': open("prompts/convert_question/PoT_2", 'r').read(),
}

en_prompts_test_one_shot = {
    'CoT': open("prompts/easy_one-shot/CoT_one-shot", 'r').read(),
    'PoT': "Let's write a Python program.",
}

en_prompts_test_0 = {
    'CoT': "Question: {query} Let's calculate step by step.",
    'PoT': "Let's write a Python program.",
}

en_prompts_test_0_1 = {
    'CoT': "Question: {query} Let's solve the question step by step.",
    'PoT': "Let's write a Python program.",
}

en_prompts_test_1 = {
    'CoT': "Below is a questions about financial calculation.\nQuestion: {query}\nPlease list all variables and "
           "required calculation formulas in the question.",
    'PoT': "Let's write a Python program.",
}

en_prompts_test_2 = {
    'CoT': "Below is a questions about financial calculation.\nQuestion: {query}\nPlease list all the numerical "
           "information and calculation formulas in the question.",
    'PoT': "Let's write a Python program.",
}

en_prompts_test_3 = {
    'CoT': "Below is a questions about financial calculation.\nQuestion: {query}\nPlease list all the numerical "
           "information in the question. Focus exclusively on the numbers given in the question, assigning each a "
           "unique variable.",
    'PoT': "Let's write a Python program.",
}

en_prompts_test_4 = {
    'CoT': "Below is a questions about financial calculation.\nQuestion: {query}\nPlease first list all the numerical "
           "information in the question. Focus exclusively on the numbers given in the question, assigning each a "
           "unique variable. Then list all calculation formulas.",
    'PoT': "Let's write a Python program.",
}

en_prompts_test_5 = {
    'CoT_1': "Below is a questions about financial calculation.\nQuestion: {query}\nPlease list all the numerical "
             "information in the question. Focus exclusively on the numbers given in the question, assigning each a "
             "unique variable. Remember not to generate any additional information!!!",
    'CoT_2': "Below is a questions about financial calculation.\nQuestion: {query}\nYou will be provided with some "
             "input variables separated by three double quotes, which have been extracted from the question."
             "\n{CoT_1}\nYour task is to proceed with your thinking in the following order: Now that you have the input"
             "variables from the question, what are the target variables mentioned in it? How should you solve for "
             "these target variables using the input variables? What intermediate variables will emerge in this "
             "process? After careful thought, please list the calculation formulas step by step. Remember not to "
             "generate any additional information and especially avoid performing calculations!!!",
    'PoT': "Below is a questions about financial calculation.\nQuestion: {query}\nYou will be provided with some "
           "input variables and step-by-step calculation formulas, separated by three double quotes.\n{CoT}\n"
           "Let's write a Python program.",
}

en_prompts_dic = {
    'CoT': "Below is a questions about financial calculation.\nQuestion:\n{query}\nPlease list the "
           "following in the specified format: variable definition, concept interpretation, "
           "calculation formula.\n\nAnswer:\n",
    'PoT': "Below is a questions about financial calculation. I have provided variable definition, "
           "concept interpretation and calculation formula.\nQuestion:\n{query}\n{CoT}\nLet's "
           "write a Python program.\n\nAnswer:\n",
    'self_correct': "Below is a Python code that you provided, but it contains error: {error}\n"
                    "{code}\nPlease correct the code.\n\nAnswer:\n",
    # 'compare': "Below is the solution process for CoT and python code, but they arrived at "
    #            "different answers.\nCoT:\n{CoT}\n\nPython code:\n{code}\nPlease help me analyze "
    #            "this and come to a final conclusion.\n\nAnswer:"
}
zh_prompts_dic = {
    'CoT': "下面是一道有关金融计算的题目。\n题目:\n{query}\n请按照以下格式列出：变量定义，概念解释，计算公式。\n\n答案:\n",
    'PoT': "下面是一道有关金融计算的题目。已经给出了变量定义、概念解释和计算公式：\n题目:\n{query}\n{CoT}\n请编写一个Python程序。\n\n答案:\n",
    'self_correct': "以下是您提供的 Python 代码，但它包含错误: {error}\n{code}\n请更正代码。\n\n答案:\n",
    # 'compare': "Below is the solution process for CoT and python code, but they arrived at "
    #            "different answers.\nCoT:\n{CoT}\n\nPython code:\n{code}\nPlease help me analyze "
    #            "this and come to a final conclusion.\n\n### Answer:"
}

en_prompts_dic_chat = {
    'CoT': "Below is a questions about financial calculation.\nQuestion:\n{query}\nPlease list the "
           "following in the specified format: variable definition, concept interpretation, "
           "calculation formula.",
    'PoT': "Below is a questions about financial calculation. I have provided variable definition, "
           "concept interpretation and calculation formula.\nQuestion:\n{query}\n{CoT}\nLet's "
           "write a Python program.",
    'self_correct': "Below is a Python code that you provided, but it contains error: {error}\n"
                    "{code}\nPlease correct the code.",
    # 'compare': "Below is the solution process for CoT and python code, but they arrived at "
    #            "different answers.\nCoT:\n{CoT}\n\nPython code:\n{code}\nPlease help me analyze "
    #            "this and come to a final conclusion.\n\nAnswer:"
}
zh_prompts_dic_chat = {
    'CoT': "下面是一道有关金融计算的题目。\n题目:\n{query}\n请按照以下格式列出：变量定义，概念解释，计算公式。",
    'PoT': "下面是一道有关金融计算的题目。已经给出了变量定义、概念解释和计算公式：\n题目:\n{query}\n{CoT}\n请编写一个Python程序。",
    'self_correct': "以下是您提供的 Python 代码，但它包含错误: {error}\n{code}\n请更正代码。",
    # 'compare': "Below is the solution process for CoT and python code, but they arrived at "
    #            "different answers.\nCoT:\n{CoT}\n\nPython code:\n{code}\nPlease help me analyze "
    #            "this and come to a final conclusion.\n\n### Answer:"
}


def chat_prompt(prompt: str):
    sys_prompt = "A chat."
    prefix = "<|im_start|>"
    suffix = "<|im_end|>\n"
    sys_format = prefix + "system\n" + sys_prompt + suffix
    user_format = prefix + "user\n" + prompt + suffix
    assistant_format = prefix + "assistant\n"
    input_text = sys_format + user_format + assistant_format
    return input_text


def drop_prompt(prompt: str):
    prefix = "<|im_start|>"
    suffix = "<|im_end|>"
    assistant_format = prefix + " assistant\n"
    temp = prompt.split(assistant_format)[1]
    output = temp.split(suffix)[0]
    return output


def extract_code(prompt: str):
    temp_list = prompt.split("```python")
    if len(temp_list) > 1:
        temp = temp_list[1]
    else:
        temp = temp_list[0]
    code = temp.split("```")[0]
    return code


if __name__ == '__main__':
    text_str = '''
<s><|im_start|> system
A chat.<|im_end|><|im_start|> user
下面是一道有关金融计算的题目。已经给出了变量定义、概念解释和计算公式：
题目:
To stimulate economic growth, since 2022, the country has repeatedly lowered bank deposit interest rates. The three-year rate was 3.15% in August 2022, reduced to 3.00% in October, and further to 2.85% in January 2023. These rates are simple/arithmetic annual interest rates. In January 2023, if Mr. Wang has a deposit of $100,000 for a fixed term of three years (note that this is a fixed-term deposit for three years), how much less interest will he earn compared to a similar three-year term deposit that if he had started in August 2022? Please use simple or arithmetic compound interest in your calculations, and do not use exponential or continuous compounding.

 1. 定义：
   - 初始金额：Principal (P)
   - 年利率：Annual Interest Rate (r)
   - 年限：Term (T)

2. 概念解释：
   - 固定期限存款（Fixed-term Deposit）：存款的金额在存款期限到期前不可提取，到期后可取。
   - 简单或算术累积利息（Simple or Arithmetic Compound Interest）：利息在每期的末日计算，并在下一期的开始日加到本金上。

3. 计算公式：
   - 固定期限存款利息（Fixed-term Deposit Interest）：I = P * r * T

4. 问题分析：
   - 2022年8月初始利率：3.15%
   - 2022年10月利率：3.00%
   - 2023年1月利率：2.85%
   - Mr. Wang的存款：100,000
   - 存款期限：3年

5. 计算过程：
   - 2022年8月利率：I1 = 100,000 * 0.0315 * 3 = 9450
   - 2022年10月利率：I2 = 100,000 * 0.0300 * 3 = 9000
   - 2023年1月利率：I3 = 100,000 * 0.0285 * 3 = 8550

6. 答案：
   - 2022年8月起的利息：9450
   - 2023年1月起的利息：8550
   - 差额：9450 - 8550 = 800

Mr. Wang在2023年1月起的固定期限存款利息将比2022年8月起的利息低800美元。
请编写一个Python程序。<|im_end|><|im_start|> assistant
 下面是一个Python程序，用于计算Mr. Wang在2023年1月起的固定期限存款利息将比2022年8月起的利息低多少：

```python
# 定义变量
initial_amount = 100000  # 初始金额
term = 3  # 年限
rates = [3.15, 3.00, 2.85]  # 年利率列表

# 计算利息
def calculate_interest(rate, term):
    return initial_amount * rate * term

# 计算2022年8月起的利息
interest_august = calculate_interest(rates[0], term)

# 计算2023年1月起的利息
interest_january = calculate_interest(rates[2], term)

# 计算差额
difference = interest_august - interest_january

print("Mr. Wang在2023年1月起的固定期限存款利息将比2022年8月起的利息低", difference, "美元。")
```

运行上述程序，输出结果为：

```
Mr. Wang在2023年1月起的固定期限存款利息将比2022年8月起的利息低 800 美元。
```

因此，Mr. Wang在2023年1月起的固定期限存款利息将比2022年8月起的利息低800美元。<|im_end|>    
    '''

    print(drop_prompt(text_str))
    print("code:\n", extract_code(drop_prompt(text_str)))
