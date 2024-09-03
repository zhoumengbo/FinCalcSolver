import torch
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from utils.chat_utils import get_gpu_memory, truncate_on_line_loop_threshold
from transformers import set_seed
set_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import os
os.environ['PYTHONHASHSEED'] = str(42)

BASE_SYSTEM_MESSAGE = """You are OpenChat, an userful AI assistant."""


def load_model(model_path, model_device):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=model_device, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def get_answer(model, tokenizer, input_str, logger, tokenizer_device):
    generation_config = GenerationConfig(
        max_new_tokens=8192, temperature=0.0, top_p=1, repetition_penalty=1.0,
        do_sample=False, use_cache=True,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        stop=["<|end_of_turn|>"]
    )
    inputs = tokenizer(input_str, return_tensors="pt", return_attention_mask=True).to(tokenizer_device)
    logger.info(get_gpu_memory())
    torch.cuda.empty_cache()
    logger.info("torch.cuda.empty_cache()")
    logger.info(get_gpu_memory())
    outputs = model.generate(**inputs, generation_config=generation_config)
    logger.info(get_gpu_memory())
    output_str = tokenizer.batch_decode(outputs)[0]
    return output_str


def chat(history, system_message, model, tokenizer, model_name, input_str, logger, tokenizer_device, is_json):
    history = history or []
    history.append([model_name, input_str, ""])

    if system_message.strip():
        messages = system_message.strip() + "\n" + \
                   "\n".join(["\n".join(["GPT4 Correct User:\n" + item[1] + "<|end_of_turn|>",
                                         "GPT4 Correct Assistant:\n" + item[2] + "<|end_of_turn|>"]) for item in history])
    else:
        messages = BASE_SYSTEM_MESSAGE + "\n" + \
                   "\n".join(["\n".join(["GPT4 Correct User:\n" + item[1] + "<|end_of_turn|>",
                                         "GPT4 Correct Assistant:\n" + item[2] + "<|end_of_turn|>"]) for item in history])
    messages = messages.rstrip("<|end_of_turn|>")
    messages = messages.rstrip()
    output_str = get_answer(model, tokenizer, messages, logger, tokenizer_device)
    answer = output_str.split("<|end_of_turn|>")[-2]
    answer = answer.strip("\nGPT4 Correct Assistant:")
    answer = truncate_on_line_loop_threshold(answer, is_json, 20, 3000)
    history[-1][2] += answer
    return history, answer, messages


if __name__ == '__main__':
    model1_name = "openchat-3.5-0106"
    model_dir = "/home/ligy/zmb_workspace/model/"
    auto_dir = "prompts/template/Auto/"
    model_device1 = 'auto'
    model_tokenizer_device = 'cuda'
    model1_path = model_dir + model1_name
    model1, tokenizer1 = load_model(model1_path, model_device1)

    question = ("The capital structure of Ricketti Enterprises, Inc., consists of 15 million shares of common stock"
                "and 1 million warrants. Each warrant gives its owner the right to purchase 1 share of common stock "
                "for an exercise price of $19. The current stock price is $25, and each warrant is worth $7. All "
                "warrant holders choose to exercise their warrants today, this will lead to the issuance of new "
                "shares, what is the new stock price?")
    history = []
    history, answer, messages = chat(history, BASE_SYSTEM_MESSAGE, model1, tokenizer1, model1_name, question, None, model_tokenizer_device, False)
    print("history: {0}".format(history))
    print("answer: {0}".format(answer))
    print("messages: {0}".format(messages))

    history, answer, messages = chat(history, "You are OpenChat, an userful AI assistant.", model1, tokenizer1,
                                     model1_name, "hello", None, model_tokenizer_device, False)
    print("history: {0}".format(history))
    print("answer: {0}".format(answer))
    print("messages: {0}".format(messages))

