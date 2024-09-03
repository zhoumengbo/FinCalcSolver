import json
from datetime import datetime
import torch
import os
from utils import chat_utils, dolphin_chat_utils, neural_chat_utils, openorca_chat_utils

from utils.chat_utils import get_gpu_memory
from utils.logger_config import LoggerConfig
import traceback
from fastapi import FastAPI, Request

app = FastAPI()
# uvicorn fastapi_server:app --host '0.0.0.0' --port 8080

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()
date = datetime.now().date()
now_time = datetime.now().time()

model_dir = "/mnt/zmb/zmb_workspace/model/"
model_name = "dolphin-2.6-mistral-7b-dpo"
# model_name = "Mistral-7B-OpenOrca"
# model_name = "neural-chat-7b-v3"
log_dir = 'outputs/server/{0}'.format(date)
os.makedirs(log_dir, exist_ok=True)
logger = LoggerConfig(log_file='{0}/{1}.log'.format(log_dir, now_time)).logger

# load models
model_device = 'auto'
model_tokenizer_device = 'cuda'
model_path = model_dir + model_name
logger.info('model load: {0}'.format(model_name))
model, tokenizer = dolphin_chat_utils.load_model(model_path, model_device)
logger.info('model started: {0}'.format(model_name))
logger.info(get_gpu_memory())


def model_chat(get_json_str):
    logger.info('run model_chat(): {0}'.format(model_name))
    get_json = json.loads(get_json_str)
    history = get_json["history"]
    system_m = get_json["system_m"]
    user_input = get_json["user_input"]
    is_json = get_json["is_json"]
    history, answer, input_all = dolphin_chat_utils.chat(history, system_m, model, tokenizer, model_name,
                                                         user_input, logger, model_tokenizer_device, is_json)

    logger.info("##### input_all #####:\n{0}".format(input_all))
    logger.info('\n##### Answer #####:\n{0}\n\n'.format(answer))
    rs = {"history": history, "answer": answer, "input_all": input_all}
    return rs


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/post")
async def json_post(request: Request):
    try:
        get_json = await request.json()
        rs = model_chat(get_json)
        return rs
    except Exception as e:
        print(''.join(traceback.format_exception(type(e), e, e.__traceback__)))
        return {"error": str(e)}
