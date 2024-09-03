import queue

import requests
import threading

from utils.cross_models_utils import *
from Create_tree_paths import *
import json
from datetime import datetime
import torch
import os
from utils import chat_utils, dolphin_chat_utils, neural_chat_utils, openorca_chat_utils
import time

from utils.chat_utils import get_gpu_memory
from utils.intercept_error import convert_full_text
from utils.logger_config import LoggerConfig
from utils.sympy_for_paths import paths_to_sympy, sympy_check, list_to_sympy

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()
print(get_gpu_memory())
date = datetime.now().date()
now_time = datetime.now().time()

model_dir = "/mnt/zmb/zmb_workspace/model"
role_dir = "prompts/model_role/"
role_dict = {"1": "financial_analyst_assistant_sample",
             "2": "corrector",
             "3": "financial_teacher",
             "4": "debater",
             "5": "teacher",
             "6": "judge"
             }
auto_dir = "prompts/template/Auto/"
dataset_file = "dataset/benchmark/set_0.txt"
save_json_dir = "outputs/json/set_0/"

model1_name = "dolphin-2.2.1-mistral-7b"
model2_name = "dolphin-2.6-mistral-7b-dpo"
dolphin_2_6_server = "http://192.168.0.109:8080/"

log_dir = 'outputs/log/{0}'.format(date)
os.makedirs(log_dir, exist_ok=True)
logger = LoggerConfig(log_file='{0}/{1}.log'.format(log_dir, now_time)).logger

# result = requests.get(dolphin_2_6_server).json()
# logger.info("Get from {0}({1}): {2}".format(model2_name, dolphin_2_6_server, result))
# model_device = 'auto'
# model_tokenizer_device = 'cuda'
model_device = 'cuda:1'
model_tokenizer_device = 'cuda:1'
model1_path = model_dir + model1_name
model, tokenizer = dolphin_chat_utils.load_model(model1_path, model_device)
logger.info(get_gpu_memory())


def main():
    Questions = []
    with open(dataset_file, 'r', encoding='utf-8') as file_in:
        for query in file_in:
            query = convert_full_text(query)
            Questions.append(query)

    time.sleep(0.5)
    question_choose = input("请选择题目编号:")
    question = Questions[int(question_choose) - 1].strip()
    q_json = json.load(open(save_json_dir + "q_{0}.json".format(question_choose), 'r', encoding='utf-8'))
    if not check_json_completeness(q_json):
        logger.error("Json is incomplete !!!")
        return
    q_json_v = {"Target Variable(s)": q_json["Target Variable(s)"], "Input Variable(s)": q_json["Input Variable(s)"]}
    q_json_f = q_json["List all formula(s)/equation(s)"]

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
    sympy_result_dict = {}
    for tree in target_builder:
        if tree.key_index > 1:
            logger.info("Formula_list: \n{0}".format(formula_list))
            logger.info("Variables_set: \n{0}".format(variables_set))
            logger.info("Input_values_dict: \n{0}".format(input_values_dict))

            target_v = tree.root.value
            all_formula_paths = extract_formula_paths(tree.root)
            result_list = []
            for formula_path in all_formula_paths:
                logger.info("===== Path =====")
                result = paths_to_sympy(formula_path, variables_set, input_values_dict, target_v, logger)
                logger.info("Result: {0}".format(result))
                result_list.append(result)

            solution_paths = extract_paths(tree.root, input_values_dict)
            logger.info("Solution_Paths: \n{0}".format(print_paths(solution_paths)))
            logger.info("Result List: \n{0}".format(result_list))

            paths_count = len(solution_paths)
            remove_sympy_not_num(result_list, solution_paths, all_formula_paths)
            # remove_sympy_error_none(result_list, solution_paths, all_formula_paths)

            if 0 in result_list and check_0_unreasonable(target_v, q_json_v):
                remove_sympy_0(result_list, solution_paths, all_formula_paths)

            if contains_positive_and_negative(result_list) and check_negative_unreasonable(target_v, q_json_v):
                remove_sympy_negative(result_list, solution_paths, all_formula_paths)

            sympy_num_paths_count = len(solution_paths)

            solution_paths_str = print_paths(solution_paths)
            logger.info("Solution_Paths after remove: \n{0}".format(solution_paths_str))
            logger.info("Target Variable: {0}, Paths Count：{1}, Sympy Num Paths Count：{2}, Result List: {3}"
                        .format(target_v, paths_count, sympy_num_paths_count, result_list))
            if len(solution_paths) > 26:
                logger.error("Sympy num Paths Count：{2} > 26, too many !!!")
                return
            letter_paths_dict = letter_paths(solution_paths)

            if len(result_list) > 0:
                sympy_result = model_cross(letter_paths_dict, question, q_json_v, all_formula_paths, variables_set,
                                           input_values_dict, target_v)
                input_values_dict[tree.root.value] = sympy_result
                sympy_result_dict[target_v] = sympy_result
                logger.info("Target Variable: {0}, Final Sympy Result: {1}".format(target_v, sympy_result))
            else:
                logger.info("Have no Sympy Result !!!")
        else:
            logger.info("Can not create a tree. Use list_to_sympy():")
            logger.info("Formula_list: {0}".format(formula_list))
            all_solutions = list_to_sympy(formula_list, variables_set, input_values_dict, logger)
            logger.info("All_solutions: {0}".format(all_solutions))
    logger.info("sympy_result_dict: {0}".format(sympy_result_dict))


def model_cross(letter_paths_dict, question, q_json_v, all_formula_paths, variables_set, input_values_dict, target_v):
    logger.info('run model_cross(): {0}, {1}'.format(model1_name, model2_name))

    if len(letter_paths_dict) > 1:
        pairs, best_choice_index = binary_choice_optimization(letter_paths_dict, question, q_json_v)
        formula_path = all_formula_paths[best_choice_index]
    else:
        formula_path = all_formula_paths[0]
    return path_step_check(formula_path, variables_set, input_values_dict, target_v)


def path_step_check(formula_path, variables_set, input_values_dict, target_v):
    sympy_result = paths_to_sympy(formula_path, variables_set, input_values_dict, target_v, logger)
    logger.info("### Optimal Path, sympy_result: {0}, formula_path: {1}".format(sympy_result, formula_path))
    return sympy_result


def binary_choice_optimization(letter_paths_dict, question, q_json_v):
    original_pairs = list(letter_paths_dict.items())
    pairs = original_pairs.copy()
    while len(pairs) > 1:
        for i in range(0, len(pairs) - 1, 2):
            chosen_pair = choose_one_path(pairs[i], pairs[i + 1], question, q_json_v)
            logger.info(f"Chosen_pair：{chosen_pair}")
            pairs.remove(pairs[i] if chosen_pair == pairs[i + 1] else pairs[i + 1])
            break
    best_choice = pairs[0]
    best_choice_index = original_pairs.index(best_choice)
    return pairs, best_choice_index


def choose_one_path(path1, path2, question, q_json_v):
    history_model1 = []
    history_model2 = []
    role_path = "prompts/model_role/debater"
    system_m = open(role_path, 'r', encoding='utf-8').read().replace("\n", " ")

    solution_2_diff = open(auto_dir + "cross/solution_2_diff.txt", 'r').read().format(
        Question=question, variable='{0}'.format(q_json_v), solution1=path1[1],
        solution2=path2[1], solution1_n=path1[0], solution2_n=path2[0])
    solution_2_choose = open(auto_dir + "cross/solution_2_choose_3.txt", 'r').read()
    solution_2_json = open(auto_dir + "cross/solution_2_json.txt", 'r').read()

    # model2
    def model2_func(history_model2, system_m, solution_2_diff, solution_2_choose, solution_2_json, result_queue):
        history_model2, _, _ = post_server(history_model2, system_m, solution_2_diff)
        history_model2, _, _ = post_server(history_model2, system_m, solution_2_choose)
        history_model2, model2_output_json, input_all = post_server(history_model2, system_m, solution_2_json)
        result_queue.put(history_model2)
        result_queue.put(model2_output_json)
        result_queue.put(input_all)

    result_queue = queue.Queue()
    model2_thread = threading.Thread(target=model2_func, args=(history_model2, system_m, solution_2_diff,
                                                               solution_2_choose, solution_2_json, result_queue))
    model2_thread.start()

    # model1
    model_name = model1_name
    history_model1, _, _ = dolphin_chat_utils.chat(history_model1, system_m, model, tokenizer, model_name,
                                                   solution_2_diff, logger, model_tokenizer_device, False)
    history_model1, _, _ = dolphin_chat_utils.chat(history_model1, system_m, model, tokenizer, model_name,
                                                   solution_2_choose, logger, model_tokenizer_device, False)
    history_model1, model1_output_json, input_all = dolphin_chat_utils.chat(history_model1, system_m, model, tokenizer,
                                                                            model_name, solution_2_json, logger,
                                                                            model_tokenizer_device, True)
    logger.info("\n##### {0} Discuss Input_all #####\n{1}".format(model1_name, input_all))
    logger.info('\n##### {0} Discuss Answer #####\n{1}\n\n'.format(model1_name, model1_output_json))

    model2_thread.join()
    history_model2 = result_queue.get()
    model2_output_json = result_queue.get()
    model2_input_all = result_queue.get()
    logger.info("\n##### {0} Discuss Input_all #####\n{1}".format(model2_name, model2_input_all))
    logger.info('\n##### {0} Discuss Answer #####\n{1}\n\n'.format(model2_name, model2_output_json))

    model1_json = json.loads(model1_output_json.strip())
    model2_json = json.loads(model2_output_json.strip())
    model1_choose = model1_json["Superior solution"].strip()
    model2_choose = model2_json["Superior solution"].strip()
    logger.info("##### Choose Result ({0}, {1}) #####\n{2}: {3}\n{4}: {5}\n".format(
        path1[0], path2[0], model1_name, model1_json, model2_name, model2_json))
    if model1_choose == model2_choose and model1_choose.strip() in [path1[0], path2[0]]:
        logger.info("##### Superior solution choice is consistent：{0}".format(model1_choose))
    else:
        logger.info("Choices are inconsistent, start a discussion ...")
        turn = 0
        while turn < 3 and model1_choose != model2_choose:
            turn += 1
            give_agent_opinion_to_model2 = open(auto_dir + "cross/give_agent_opinion.txt", 'r').read().format(
                opinion=model1_json["Reason"])
            history_model2, answer, _ = post_server(history_model2[:-1], system_m, give_agent_opinion_to_model2)
            history_model2, model2_output_json, model2_input = post_server(history_model2, system_m, solution_2_json)
            logger.info('\n##### {0} Discuss Input #####\n{1}\n\n'.format(model2_name, model2_input))
            logger.info('\n##### {0} Discuss Answer #####\n{1}\n\n'.format(model2_name, model2_output_json))
            model2_json = json.loads(model2_output_json.strip())

            give_agent_opinion_to_model1 = open(auto_dir + "cross/give_agent_opinion.txt", 'r').read().format(
                opinion=model2_json["Reason"])
            history_model1, answer, input_all = dolphin_chat_utils.chat(
                history_model1[:-1], system_m, model, tokenizer, model_name, give_agent_opinion_to_model1, logger,
                model_tokenizer_device, False)
            history_model1, model1_output_json, model1_input = dolphin_chat_utils.chat(
                history_model1, system_m, model, tokenizer, model_name, solution_2_json, logger, model_tokenizer_device, True)
            logger.info('\n##### {0} Discuss Input #####\n{1}\n\n'.format(model1_name, model1_input))
            logger.info('\n##### {0} Discuss Answer #####\n{1}\n\n'.format(model1_name, model1_output_json))
            model1_json = json.loads(model1_output_json.strip())
            model1_choose = model1_json["Superior solution"].strip()
            model2_choose = model2_json["Superior solution"].strip()

    if model1_choose == model2_choose:
        judge_choose = model1_choose
        logger.info("##### After discussion, the final choice is: {0}\n".format(judge_choose))
    else:
        logger.info("The choices are still inconsistent after discussion, the final judgment is made ...")
        history_judge = []
        role_path = "prompts/model_role/judge"
        system_m = open(role_path, 'r', encoding='utf-8').read().replace("\n", " ")
        solution_judge_1 = open(auto_dir + "cross/solution_judge_1.txt", 'r').read().format(
            Question=question.strip(), solution1=path1[1], solution2=path2[1], agent1=str(model1_json),
            agent2=str(model2_json))
        solution_judge_2 = open(auto_dir + "cross/solution_judge_2.txt", 'r').read()
        history_judge, answer, input_all = dolphin_chat_utils.chat(
            history_judge, system_m, model, tokenizer, model_name, solution_judge_1, logger, model_tokenizer_device, False)
        history_judge, judge_output_json, judge_input = dolphin_chat_utils.chat(
            history_judge, system_m, model, tokenizer, model_name, solution_judge_2, logger, model_tokenizer_device, True)
        logger.info('\n##### {0} Judge Input #####\n{1}\n\n'.format(model1_name, judge_input))
        logger.info('\n##### {0} Judge Answer #####\n{1}\n\n'.format(model1_name, judge_output_json))
        judge_json = json.loads(judge_output_json.strip())
        judge_choose = judge_json["Superior solution"].strip()
        logger.info("##### After judge, the final choice is: {0}\n".format(judge_choose))
    if judge_choose == path1[0]:
        return path1
    elif judge_choose == path2[0]:
        return path2
    else:
        return None


def check_0_unreasonable(target_v, q_json_v):
    role_path = "prompts/model_role/financial_analyst_assistant_sample"
    system_m = open(role_path, 'r', encoding='utf-8').read().replace("\n", " ")
    history_check = []
    check_0_prompt = open(auto_dir + "check/check_0.txt", 'r').read().format(t_v=target_v, q_json_v=q_json_v)
    check_0_json_prompt = open(auto_dir + "check/check_0_json.txt", 'r').read()

    history_check, _, _ = dolphin_chat_utils.chat(history_check, system_m, model, tokenizer, model1_name,
                                                  check_0_prompt, logger, model_tokenizer_device, False)
    history_check, check_0_output, check_0_input = dolphin_chat_utils.chat(history_check, system_m, model, tokenizer,
                                                                           model1_name, check_0_json_prompt, logger,
                                                                           model_tokenizer_device, True)
    logger.info('\n##### {0} Check 0 Input #####\n{1}\n\n'.format(model1_name, check_0_input))
    logger.info('\n##### {0} Check 0 Answer #####\n{1}\n\n'.format(model1_name, check_0_output))
    check_0_json = json.loads(check_0_output.strip())
    check_0_choose = check_0_json["It is reasonable for this target variable to be zero"].strip()
    if check_0_choose == "No":
        return True
    else:
        return False


def check_negative_unreasonable(target_v, q_json_v):
    role_path = "prompts/model_role/financial_analyst_assistant_sample"
    system_m = open(role_path, 'r', encoding='utf-8').read().replace("\n", " ")
    history_check = []
    check_symbol_prompt = open(auto_dir + "check/check_symbol.txt", 'r').read().format(t_v=target_v, q_json_v=q_json_v)
    check_symbol_json_prompt = open(auto_dir + "check/check_symbol_json.txt", 'r').read()

    history_check, _, _ = dolphin_chat_utils.chat(history_check, system_m, model, tokenizer, model1_name,
                                                  check_symbol_prompt, logger, model_tokenizer_device, False)
    history_check, check_symbol_output, check_symbol_input = dolphin_chat_utils.chat(history_check, system_m, model,
                                                                                     tokenizer,  model1_name,
                                                                                     check_symbol_json_prompt, logger,
                                                                                     model_tokenizer_device, True)
    logger.info('\n##### {0} Check Symbol Input #####\n{1}\n\n'.format(model1_name, check_symbol_input))
    logger.info('\n##### {0} Check Symbol Answer #####\n{1}\n\n'.format(model1_name, check_symbol_output))
    check_symbol_json = json.loads(check_symbol_output.strip())
    check_symbol_choose = check_symbol_json["It is reasonable for this target variable to be negative"].strip()
    if check_symbol_choose == "No":
        return True
    else:
        return False


def contains_positive_and_negative(lst):
    has_positive = any(x > 0 for x in lst)
    has_negative = any(x < 0 for x in lst)
    return has_positive and has_negative


def model_chat():
    logger.info('run model_chat(): {0}, {1}'.format(model1_name, model2_name))

    # define roles
    roles_initialized = False
    mode1_system_m = ""
    mode2_system_m = ""
    role1 = ""
    role2 = ""

    history = []
    chat_num = 1
    while True:
        time.sleep(2)
        if not roles_initialized:
            model1_role, mode1_system_m = role_choose(model1_name)
            if model1_role is None:
                logger.error("Error in role_choose: ", mode1_system_m)
                continue
            model2_role, mode2_system_m = role_choose(model2_name)
            if model2_role is None:
                logger.error("Error in role_choose: ", mode2_system_m)
                continue
            role1 = '{0}({1})'.format(model1_name, model1_role)
            role2 = '{0}({1})'.format(model2_name, model2_role)
            roles_initialized = True
        continue_chat = input(
            "1: {0}, 2: {1}, 3: {2}, 4: {3}, 5: {4}, 6: Exit, 请选择:".format(
                "继续聊天", "清空历史", "转换角色", "清空历史并转换角色", "清空上一步聊天"))
        if continue_chat == "6":
            break
        elif continue_chat == "5":
            history = history[:-1]
            continue
        elif continue_chat in ["2", "4"]:
            history.clear()
            chat_num += 1
            if continue_chat == "4":
                roles_initialized = False
                continue
        elif continue_chat == "3":
            roles_initialized = False
            continue
        elif continue_chat == "1":
            time.sleep(0.2)
        else:
            logger.error("Continue_chat number not invalid: {0}".format(continue_chat))
            continue
        model_choose = input("1: {0}, 2: {1}, 请选择输入的模型:".format(role1, role2))
        input_choose = input("1: input-text-box, 2: /prompts/test, 请选择输入的方式:")
        if input_choose == "1":
            user_input = chat_utils.get_input()
        elif input_choose == "2":
            prompt_file = open("prompts/test", 'r')
            user_input = prompt_file.read()
        else:
            logger.error("Input_choose number not invalid: {0}".format(input_choose))
            continue
        if model_choose == "1":
            role = role1
            history, answer, input_all = dolphin_chat_utils.chat(history, mode1_system_m, model, tokenizer,
                                                                 model1_name, user_input, logger,
                                                                 model_tokenizer_device, True)
        elif model_choose == "2":
            role = role2
            history, answer, input_all = post_server(history, mode2_system_m, user_input)
        else:
            logger.error("model_choose not invalid: {0}".format(model_choose))
            continue
        logger.info("##### {0} #####\n##### input_all #####:\n{1}".format(role, input_all))
        logger.info('\n##### Answer #####:\n{0}\n\n'.format(answer))


def post_server(history, mode_system_m, user_input):
    input_json = {"history": history, "system_m": mode_system_m, "user_input": user_input}
    result = requests.post(dolphin_2_6_server + "post", json=json.dumps(input_json)).json()
    history = result["history"]
    answer = result["answer"]
    input_all = result["input_all"]
    return history, answer, input_all


def role_choose(model_name: str):
    prompt = ", ".join(["{0}: {1}".format(key, value) for key, value in role_dict.items()])
    prompt = prompt + ", 请选择模型{}的角色: ".format(model_name)
    model_role_choose = input(prompt)
    print("您选择的角色是: " + role_dict.get(model_role_choose, "无效的选择"))
    try:
        model_role = role_dict[model_role_choose]
        role_path = role_dir + model_role
        model_system_message = open(role_path, 'r', encoding='utf-8').read().replace("\n", " ")
    except Exception as e:
        return None, e
    else:
        return model_role, model_system_message


if __name__ == '__main__':
    main()
