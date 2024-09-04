from datetime import datetime
import torch
import os
from utils import chat_utils, dolphin_chat_utils, arithmo_chat_utils, openorca_chat_utils, neural_chat_utils
import time
from utils.logger_config import LoggerConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()
date = datetime.now().date()
now_time = datetime.now().time()

model_dir = "/mnt/zmb/zmb_workspace/model/"
role_dir = "prompts/model_role/"
role_dict = {"1": "financial_analyst_assistant_sample",
             "2": "corrector",
             "3": "debater_sample",
             "4": "debater",
             "5": "teacher",
             "6": "judge"
             }


def main():
    model1_name = "Mistral-7B-OpenOrca"
    model2_name = "dolphin-2.2.1-mistral-7b"
    model3_name = ""
    log_dir = 'outputs/cross_validation/{0}'.format(date)
    os.makedirs(log_dir, exist_ok=True)
    logger = LoggerConfig(log_file='{0}/{1}.log'.format(log_dir, now_time)).logger
    model_chat(model1_name, model2_name, model3_name, logger)


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


def print_history(data):
    string_representation = ["\n\n### {}\n### Input:\n{}\n\n### Output:\n{}\n".format(name, inp, out) for name, inp, out
                             in data]
    return "\n".join(string_representation)


def model_chat(model1_name: str, model2_name: str, model3_name: str, logger):
    logger.info('run model_chat(): {0}, {1}'.format(model1_name, model2_name))

    # load models
    openorca_chat_utils.model_device = 'cpu'
    openorca_chat_utils.tokenizer_device = 'cpu'
    dolphin_chat_utils.model_device = 'auto'
    dolphin_chat_utils.tokenizer_device = 'cuda'
    neural_chat_utils.model_device = 'cpu'
    neural_chat_utils.tokenizer_device = 'cpu'
    model1_path = model_dir + model1_name
    model2_path = model_dir + model2_name
    model3_path = model_dir + model3_name
    # model1, tokenizer1 = openorca_chat_utils.load_model(model1_path)
    model1, tokenizer1 = None, None
    model2, tokenizer2 = dolphin_chat_utils.load_model(model2_path)
    # model3, tokenizer3 = neural_chat_utils.load_model(model3_path)

    # define roles
    roles_initialized = False
    mode1_system_message = ""
    mode2_system_message = ""
    role1 = ""
    role2 = ""

    history = []
    chat_num = 1
    while True:
        time.sleep(2)
        if not roles_initialized:
            model1_role, mode1_system_message = role_choose(model1_name)
            if model1_role is None:
                logger.error("Error in role_choose: ", mode1_system_message)
                continue
            model2_role, mode2_system_message = role_choose(model2_name)
            if model2_role is None:
                logger.error("Error in role_choose: ", mode2_system_message)
                continue
            role1 = '{0}({1})'.format(model1_name, model1_role)
            role2 = '{0}({1})'.format(model2_name, model2_role)
            roles_initialized = True
        continue_chat = input("1: {0}, 2: {1}, 3: {2}, 4: {3}, 5: Exit, 请选择:".format("继续聊天", "清空历史", "转换角色", "清空历史并转换角色"))
        if continue_chat == "5":
            break
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
        model_choose = input("1: {0}, 2: {1}, 请选择输入的模型:".format(model1_name, model2_name))
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
            history, answer, input_all = openorca_chat_utils.chat(history, mode1_system_message, model1, tokenizer1,
                                                                  model1_name, user_input)
        elif model_choose == "2":
            history, answer, input_all = dolphin_chat_utils.chat(history, mode2_system_message, model2, tokenizer2,
                                                                 model2_name, user_input)
        else:
            logger.error("model_choose not invalid: {0}".format(model_choose))
            continue
        logger.info("\n##### input_all #####:\n{0}".format(input_all))
        logger.info('\n##### Answer #####:\n{0}\n\n'.format(answer))
    save_path = "outputs/chat/{0}/{1}-{2}-{3}.docx".format(date, role1, role2, now_time)


if __name__ == '__main__':
    main()
