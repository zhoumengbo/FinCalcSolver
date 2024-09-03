import torch
from datetime import datetime

from docx import Document

from Create_tree_paths import *
from utils.chat_utils import get_gpu_memory
from utils.cross_models_utils import *
from utils.create_json_utils import *
from utils import dolphin_chat_utils

from utils.docx_utils import add_title, add_para, add_para_highlight
from utils.intercept_error import convert_full_text
from utils.logger_config import LoggerConfig
from utils.sympy_for_paths import sympy_check, paths_to_sympy

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()
date = datetime.now().date()
now_time = datetime.now().time()

model1_name = "dolphin-2.2.1-mistral-7b"
# model1_name = "Mistral-7B-OpenOrca"
model_dir = "/mnt/zmb/zmb_workspace/model/"

auto_dir = "prompts/template/Auto/"
model_device1 = 'auto'
model_tokenizer_device = 'cuda'
model1_path = model_dir + model1_name


def main():
    log_dir = 'outputs/test_create_json/dolphin-2.2.1/{0}'.format(date)
    save_docx_dir = "outputs/test_create_json/dolphin-2.2.1/"
    os.makedirs(log_dir, exist_ok=True)
    logger = LoggerConfig(log_file='{0}/{1}.log'.format(log_dir, now_time)).logger
    logger.info(get_gpu_memory())
    model1, tokenizer1 = dolphin_chat_utils.load_model(model1_path, model_device1)

    question = ("The capital structure of Ricketti Enterprises, Inc., consists of 15 million shares of common stock "
                "and 1 million warrants. Each warrant gives its owner the right to purchase 1 share of common stock "
                "for an exercise price of $19. The current stock price is $25, and each warrant is worth $7. All "
                "warrant holders choose to exercise their warrants today, this will lead to the issuance of new "
                "shares, what is the new stock price?")
    question = convert_full_text(question).strip()
    os.makedirs(save_docx_dir, exist_ok=True)
    save_json_file = save_docx_dir + "result.docx"
    json_log_doc = Document()
    json_log_doc.add_heading('已采纳题目结果', 0)
    q_json = create_json(model1_name, question, model1, tokenizer1, model_tokenizer_device, logger,
                         save_json_file, json_log_doc)
    print(q_json)
    json_log_doc.save(save_json_file)


def create_json(model_name, question, model, tokenizer, tokenizer_device, logger, save_json_file, log_doc):
    history = []
    role_path = "prompts/model_role/financial_analyst_assistant_sample"
    system_m = open(role_path, 'r', encoding='utf-8').read().replace("\n", " ")

    # s1:提取目标变量，转化JSON
    extract_target_v_prompt = open(auto_dir + "s1/1.txt", 'r').read().format(
        Question=question.strip())
    history, answer, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                          extract_target_v_prompt, logger, tokenizer_device, False)

    target_v2json_prompt = open(auto_dir + "s1/2.txt", 'r').read()
    history, target_v_json_str, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                                     target_v2json_prompt, logger, tokenizer_device,
                                                                     True)
    logger.info("\n##### s1 input_all #####:\n{0}".format(input_all))
    logger.info('\n##### s1 Answer #####:\n{0}\n\n'.format(target_v_json_str))
    target_v_json_str = format_variable_name(target_v_json_str)
    add_title(log_doc, "s1:提取目标变量，转化JSON")
    add_para(log_doc, "\n##### s1 Input_all #####:\n\n{0}".format(input_all))
    add_para_highlight(log_doc, '\n##### s1 Answer #####:\n\n{0}\n\n'.format(target_v_json_str))
    history.clear()

    # s2:提取输入变量，转化JSON
    extract_target_v_prompt = open(auto_dir + "s2/1.txt", 'r').read().format(Question=question.strip())
    history, answer, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                          extract_target_v_prompt, logger, tokenizer_device, False)

    extract_target_v_prompt = open(auto_dir + "s2/2.txt", 'r').read()
    history, answer, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                          extract_target_v_prompt, logger, tokenizer_device, False)

    extract_target_v_prompt = open(auto_dir + "s2/3.txt", 'r').read()
    history, answer, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                          extract_target_v_prompt, logger, tokenizer_device, False)

    input_v2json_prompt = open(auto_dir + "s2/4.txt", 'r').read()
    history, input_v_json_str, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                                    input_v2json_prompt, logger, tokenizer_device, True)
    logger.info("\n##### s2 input_all #####:\n{0}".format(input_all))
    logger.info('\n##### s2 Answer #####:\n{0}\n\n'.format(input_v_json_str))
    input_v_json_str = format_variable_name(input_v_json_str)
    add_title(log_doc, "s2:提取输入变量，转化JSON")
    add_para(log_doc, "\n##### s2 Input_all #####:\n\n{0}".format(input_all))
    add_para_highlight(log_doc, '\n##### s2 Answer #####:\n\n{0}\n\n'.format(input_v_json_str))
    history.clear()

    # s3:列出目标、输入、中间变量相关公式，提取输入变量并转化JSON
    # 在获得目标变量和输入变量的JSON后，第一步，列出目标变量相关公式
    target_v_to_f_prompt = open(auto_dir + "s3/2/1.txt", 'r').read().format(
        Question=question.strip(), Target_json=target_v_json_str.strip(), Input_json=input_v_json_str.strip())
    history, f3_1, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                        target_v_to_f_prompt, logger, tokenizer_device, False)
    logger.info("\n##### s3-1 input_all #####:\n{0}".format(input_all))
    logger.info('\n##### s3-1 Answer #####:\n{0}\n\n'.format(f3_1))

    # 第二步，列出输入变量相关公式
    input_v_to_f_prompt = open(auto_dir + "s3/2/2.txt", 'r').read()
    history, f3_2, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                        input_v_to_f_prompt, logger, tokenizer_device, False)
    logger.info('\n##### s3-2 Answer #####:\n{0}\n\n'.format(f3_2))

    # 第三步，将公式带入实际问题，并定义中间变量
    f_to_q_prompt = open(auto_dir + "s3/2/3.txt", 'r').read()
    history, f3_3, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                        f_to_q_prompt, logger, tokenizer_device, False)
    logger.info('\n##### s3-3 Answer #####:\n{0}\n\n'.format(f3_3))

    # 第四步，列出中间变量的公式
    inter_v_to_f_prompt = open(auto_dir + "s3/2/4.txt", 'r').read()
    history, f3_4, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                        inter_v_to_f_prompt, logger, tokenizer_device, False)
    logger.info('\n##### s3-4 Answer #####:\n{0}\n\n'.format(f3_4))

    # 第五步，中间变量转化为JSON
    inter_v_to_json_prompt = open(auto_dir + "s3/2/5.txt", 'r').read()
    history, inter_v_json_str, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                                    inter_v_to_json_prompt, logger, tokenizer_device,
                                                                    True)
    logger.info('\n##### s3-5 Answer #####:\n{0}\n\n'.format(inter_v_json_str))
    add_title(log_doc, "s3:列出目标、输入、中间变量相关公式，提取输入变量并转化JSON")
    add_para(log_doc, "\n##### s3 Input_all #####:\n\n{0}".format(input_all))
    add_para_highlight(log_doc, '\n##### s3 Answer #####:\n\n{0}\n\n'.format(inter_v_json_str))
    history.clear()

    # s4:提取相关公式并转化JSON
    S4_prompt = (open(auto_dir + "s4/one-shot.txt", 'r').read() + open(auto_dir + "s4/f_to_json.txt", 'r').read()
                 .format(f3_1=f3_1.strip(), f3_2=f3_2.strip(), f3_3=f3_3.strip(), f3_4=f3_4.strip()))
    history, f_json_str, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                              S4_prompt, logger, tokenizer_device, True)
    logger.info("\n##### s4 input_all #####:\n{0}".format(input_all))
    logger.info('\n##### s4 Answer #####:\n{0}\n\n'.format(f_json_str))
    output_json = merge_json_strings(target_v_json_str, input_v_json_str, inter_v_json_str, f_json_str)
    add_title(log_doc, "s4:提取相关公式并转化JSON")
    add_para(log_doc, "\n##### s4 Input_all #####:\n\n{0}".format(input_all))
    add_para_highlight(log_doc, '\n##### s4 Answer #####:\n\n{0}\n\n'.format(f_json_str))

    result_s4 = save_json(output_json, save_json_file)
    logger.info('\n##### s4_output_json #####:\n{0}\nsave_json_result: {1}\n'.format(output_json, result_s4))
    history.clear()

    # s5:公式与变量进行适配
    add_title(log_doc, "s5:公式与变量进行适配")
    formula_list = format_formulas(f_json_str)
    if not isinstance(formula_list, list):
        logger.info(formula_list)
        return output_json

    formula_dict = {}
    index = 1
    logger.info('\n##### format_formulas: #####:\n{0}\n\n'.format(formula_list))
    variable_dict, target_v, input_v = extract_variable_name(target_v_json_str, input_v_json_str, inter_v_json_str)
    logger.info("Variable_dict: {0}".format(variable_dict))
    for formula in formula_list:
        formula_variable_list = extract_variables(formula)
        logger.info("Formula: {0}".format(formula))
        logger.info("Formula_variable_list: {0}".format(formula_variable_list))
        if not all(elem in variable_dict.keys() for elem in formula_variable_list):
            logger.info("Variables are not compatible ！！！")
            s5_1_prompt = (open(auto_dir + "s5/1.txt", 'r').read()
                           .format(Question=question.strip(), v_dict=str(variable_dict), formula=formula))
            history, answer, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                                  s5_1_prompt, logger, tokenizer_device, False)
            s5_2_prompt = (open(auto_dir + "s5/2.txt", 'r').read().format(formula=formula))
            history, answer, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                                  s5_2_prompt, logger, tokenizer_device, False)
            s5_3_prompt = (open(auto_dir + "s5/3.txt", 'r').read())
            history, f5, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                              s5_3_prompt, logger, tokenizer_device, True)
            logger.info("\n##### s5 input_all #####:\n{0}".format(input_all))
            add_para_highlight(log_doc, "\n##### Formula: {0} #####\n\n".format(formula))
            add_para(log_doc, "\n##### Input_all #####:\n\n{0}".format(input_all))
            add_para_highlight(log_doc, '\n##### Answer #####:\n\n{0}\n\n'.format(f5))

            f5_json = extract_json_from_str(f5)
            if f5_json is not None:
                split_formula = f5_json["new_formula"].split("=")
                if len(split_formula) == 2:
                    formula = f5_json["new_formula"]
                elif len(split_formula) > 2:
                    formula = "{0} = {1}".format(split_formula[-2].strip(), split_formula[-1].strip())
            history.clear()
        logger.info("Finally Formula: {0}".format(formula))
        formula_dict["eq{0}".format(index)] = formula
        index += 1
    formula_json = {"List all formula(s)/equation(s)": formula_dict}
    add_para_highlight(log_doc, '\n##### s5 Answer #####:\n{0}\n\n'.format(formula_json))
    output_json = merge_json_strings(target_v_json_str, input_v_json_str, inter_v_json_str, formula_json)
    result_s5 = save_json(output_json, save_json_file)
    logger.info('\n##### s5_output_json #####:\n{0}\nsave_json_result: {1}\n'.format(output_json, result_s5))
    history.clear()

    # 如果有数值解，则不进行s6；否则循环每个输入变量，直到有数值解
    add_title(log_doc, "s6:如果有数值解，则不进行s6；若沒有，双变量循环（每个目标变量和输入变量），直到有数值解")
    if not check_json_completeness(output_json):
        return output_json
    while not check_num_answer(output_json, logger) and input_v:
        # s6:双变量抽取公式: 目标变量和输入变量
        i_v = input_v.pop(0)
        logger.info("##### No num answer !!! Double variable get formula #####\ntarget_v: {0}, input_v: {1}"
                    .format(target_v, i_v))
        s6_formula_dict = {}
        index = 1
        s6_1_prompt = (open(auto_dir + "s6/1.txt", 'r').read()
                       .format(Question=question.strip(), v_dict=str(variable_dict)))
        history, answer, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                              s6_1_prompt, logger, tokenizer_device, False)
        for t_v in target_v:
            history = history[:1]
            s6_2_prompt = (open(auto_dir + "s6/2.txt", 'r').read().format(t_v=t_v))
            history, answer, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                                  s6_2_prompt, logger, tokenizer_device, False)
            s6_3_prompt = (open(auto_dir + "s6/3.txt", 'r').read().format(t_v=t_v, i_v=i_v))
            history, answer, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                                  s6_3_prompt, logger, tokenizer_device, False)
            s6_4_prompt = (open(auto_dir + "s6/4.txt", 'r').read().format(v_n=str(list(variable_dict.keys()))))
            history, answer, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                                  s6_4_prompt, logger, tokenizer_device, False)
            s6_5_prompt = (open(auto_dir + "s6/5.txt", 'r').read())
            history, f6, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                              s6_5_prompt, logger, tokenizer_device, True)
            logger.info("##### s6_{0}_{1} input_all #####:\n{2}".format(t_v, i_v, input_all))
            logger.info("##### s6_{0}_{1} answer #####:\n{2}".format(t_v, i_v, f6))
            add_para_highlight(log_doc, "\n##### 目标变量：{0}, 输入变量：{1} #####:\n".format(t_v, i_v))
            add_para(log_doc, "\n##### Input_all #####:\n\n{0}".format(input_all))
            add_para_highlight(log_doc, '\n##### Answer #####:\n\n{0}\n\n'.format(f6))

            f6_json = extract_json_from_str(f6)
            if f6_json is not None:
                formulas = extract_root_elements(f6_json)
                for formula in formulas:
                    split_formula = formula.split("=")
                    if len(split_formula) > 2:
                        formula = ("{0} = {1}".format(split_formula[-2].strip(), split_formula[-1].strip()))
                    elif len(split_formula) < 2:
                        continue
                    s6_formula_dict["s6_{0}__{1}_eq{2}".format(t_v, i_v, index)] = formula
                    index += 1
            logger.info("##### s6_{0}_{1} formula_dict #####: \n{2}".format(t_v, i_v, s6_formula_dict))
        formula_dict.update(s6_formula_dict)
        formula_json = {"List all formula(s)/equation(s)": formula_dict}
        add_para(log_doc, '\n##### New Formulas #####:\n\n{0}\n\n'.format(formula_json))
        output_json = merge_json_strings(target_v_json_str, input_v_json_str, inter_v_json_str, formula_json)
        history.clear()

    result_s6 = save_json(output_json, save_json_file)
    logger.info('\n##### s6_output_json #####:\n{0}\nsave_json_result: {1}\n'.format(output_json, result_s6))
    return output_json


def check_num_answer(q_json, logger):
    q_json_f = q_json["List all formula(s)/equation(s)"]
    if len(q_json_f) == 0:
        return False
    origin_formula_list = []
    for key, value in q_json_f.items():
        formula = check_and_correct_parentheses(value)
        formula = replace_multiple_spaces_with_underscores(formula)
        origin_formula_list.append(formula)
    origin_variables_set = set()
    for formula in origin_formula_list:
        origin_variables_set.update(extract_variables(formula))

    formula_list = sympy_check(origin_formula_list, origin_variables_set, logger)

    variables_set = set()
    for formula in formula_list:
        variables_set.update(extract_variables(formula))
    input_values_dict = {}
    for variable in q_json["Input Variable(s)"]:
        if str_is_number(variable["Numerical Value"]):
            input_values_dict[variable["Variable Name"].replace("'", "")] = variable["Numerical Value"]
    target_builder = create_tree_and_paths(logger, q_json, formula_list)
    for tree in target_builder:
        target_result_is_num = False
        if tree.key_index > 1:
            logger.info("Formula_list: \n{0}".format(formula_list))
            logger.info("Variables_set: \n{0}".format(variables_set))
            logger.info("Input_values_dict: \n{0}".format(input_values_dict))

            target_v = tree.root.value
            result_num = None
            all_formula_paths = extract_formula_paths(tree.root)
            if not isinstance(all_formula_paths, list):
                logger.info("Extract formula paths from tree, " + all_formula_paths)
                continue

            logger.info("Sympy Paths Count：{0}".format(len(all_formula_paths)))
            if len(all_formula_paths) > 200:
                logger.info("Sympy Paths Count：{0} > 200, too many !!!".format(len(all_formula_paths)))
                continue
            for formula_path in all_formula_paths:
                logger.info("===== Path =====")
                result = paths_to_sympy(formula_path, variables_set, input_values_dict, target_v, logger)
                logger.info("Result: {0}".format(result))
                if str_is_number(result):
                    target_result_is_num = True
                    result_num = result
            if target_result_is_num:
                input_values_dict[tree.root.value] = result_num
        if not target_result_is_num:
            return False
    return True


if __name__ == '__main__':
    main()
