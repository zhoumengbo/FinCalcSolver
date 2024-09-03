import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_device = ''
tokenizer_device = ''


def load_model(model_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=model_device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def get_answer(model, tokenizer, input_str):
    inputs_ft = tokenizer(input_str, return_tensors="pt").to(tokenizer_device)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    generated_ids = model.generate(**inputs_ft,
                                   max_new_tokens=2048,
                                   temperature=0.0,
                                   top_p=1,
                                   pad_token_id=tokenizer.pad_token_id
                                   )
    output_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_str


def chat(history, system_message, model, tokenizer, model_name, input_str):
    history = history or []
    history.append([model_name, input_str, ""])
    messages = system_message.strip() + "\n".join(["\n".join([item[1], "Answer:\n" + item[2]])for item in history])
    output_str = get_answer(model, tokenizer, messages)
    answer = output_str.split("Answer:")[1]
    answer = answer.rstrip()
    history[-1][2] += answer
    return history, answer, messages
