import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.chat_utils import get_gpu_memory

BASE_SYSTEM_MESSAGE = """I carefully provide accurate, factual, thoughtful, nuanced answers and am brilliant at reasoning. 
I am an assistant who thinks through their answers step-by-step to be sure I always get the right answer. 
I think more clearly if I write out my thought process in a scratchpad manner first; therefore, I always explain background context, assumptions, and step-by-step thinking BEFORE trying to answer or solve anything."""


def load_model(model_path, model_device):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=model_device, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def chat(history, system_message, model, tokenizer, model_name, input_str, logger, tokenizer_device):
    history = history or []
    history.append([model_name, input_str, ""])
    if system_message.strip():
        messages = "### System:\n" + system_message.strip() + "\n".join(["\n".join(
            ["### User:\n" + item[1], "\n### Assistant:\n" + item[2]]) for item in history])
    else:
        messages = "### System:\n" + BASE_SYSTEM_MESSAGE + "\n".join(["\n".join(
            ["### User:\n" + item[1], "\n### Assistant:\n" + item[2]]) for item in history])

    inputs = tokenizer.encode(messages, return_tensors="pt", add_special_tokens=False).to(tokenizer_device)
    logger.info(get_gpu_memory())
    torch.cuda.empty_cache()
    logger.info("torch.cuda.empty_cache()")
    logger.info(get_gpu_memory())
    outputs = model.generate(inputs, max_length=4096, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("### Assistant:\n")[-1]
    history[-1][2] += answer
    return history, answer, messages

