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
os.environ['PYTHONHASHSEED']= str(42)

BASE_SYSTEM_MESSAGE = """I carefully provide accurate, factual, thoughtful, nuanced answers and am brilliant at reasoning. 
I am an assistant who thinks through their answers step-by-step to be sure I always get the right answer. 
I think more clearly if I write out my thought process in a scratchpad manner first; therefore, I always explain background context, assumptions, and step-by-step thinking BEFORE trying to answer or solve anything."""


def load_model(model_path, model_device):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=model_device, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def get_answer(model, tokenizer, input_str, logger, tokenizer_device):
    generation_config = GenerationConfig(
        max_new_tokens=8192, temperature=0.0, top_p=1, repetition_penalty=1.0,
        do_sample=False, use_cache=True,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        stop=["</s>", "<|im_end|>"]
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
        messages = "<|im_start|> " + "system:\n" + system_message.strip() + "<|im_end|>\n" + \
                   "\n".join(["\n".join(["<|im_start|> " + "user:\n" + item[1] + "<|im_end|>",
                                         "<|im_start|> assistant:\n" + item[2] + "<|im_end|>"])
                              for item in history])
    else:
        messages = "<|im_start|> " + "system:\n" + BASE_SYSTEM_MESSAGE + "<|im_end|>\n" + \
                   "\n".join(["\n".join(["<|im_start|> " + "user:\n" + item[1] + "<|im_end|>",
                                         "<|im_start|> assistant:\n" + item[2] + "<|im_end|>"])
                              for item in history])
    # strip the last `<|end_of_turn|>` from the messages
    messages = messages.rstrip("<|im_end|>")
    # remove last space from assistant, some models output a ZWSP if you leave a space
    messages = messages.rstrip()

    output_str = get_answer(model, tokenizer, messages, logger, tokenizer_device)
    answer = output_str.split("<|im_start|>  assistant:")[-1]
    answer = answer.rstrip("<|im_end|>")
    answer = truncate_on_line_loop_threshold(answer, is_json, 20, 3000)
    history[-1][2] += answer
    return history, answer, messages
