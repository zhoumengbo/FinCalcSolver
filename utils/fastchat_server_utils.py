import uuid
import requests
from utils.create_json_utils import *
from utils.model_adapter import get_conversation_template

INPUT_CHAR_LEN_LIMIT = int(os.getenv("FASTCHAT_INPUT_CHAR_LEN_LIMIT", 12000))
WORKER_API_TIMEOUT = int(os.getenv("FASTCHAT_WORKER_API_TIMEOUT", 100))


class State:
    def __init__(self, model_name, system_m, worker_addr):
        self.conv = get_conversation_template(model_name)
        self.conv.system_message = system_m
        self.conv_id = uuid.uuid4().hex
        self.skip_next = False
        self.model_name = model_name
        self.worker_addr = worker_addr

    def dict(self):
        base = self.conv.dict()
        base.update(
            {
                "conv_id": self.conv_id,
                "model_name": self.model_name,
            }
        )
        return base


def post_server(state, text):
    conv, model_name, worker_addr = state.conv, state.model_name, state.worker_addr
    text = text[:INPUT_CHAR_LEN_LIMIT]
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    prompt = conv.get_prompt()
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "top_p": 1,
        "max_new_tokens": 8192,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }

    # Stream output
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers={"User-Agent": "FastChat Client"},
        json=gen_params,
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )
    conv.update_last_message("▌")

    output = ""
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            output = data["text"].strip()
    conv.update_last_message(output)
    return prompt, output


if __name__ == '__main__':
    model_name = "openchat-3.5-0106"
    fast_chat_server = "http://localhost:21001/"

    input_json = {"model": model_name}
    result = requests.post(fast_chat_server + "get_worker_address", json={"model": model_name})
    worker_addr = result.json()["address"]
    print("Get worker_addr from {0}({1}): {2}".format(model_name, fast_chat_server, worker_addr))

    role_path = "prompts/model_role/financial_analyst_assistant_sample"
    system_m = open(role_path, 'r', encoding='utf-8').read().replace("\n", " ")

    state = State(model_name, system_m, worker_addr)
    print(post_server(state, "hello"))
    print("post 1 finish")
    print(post_server(state, "what are you doing now?"))
    print("post 2 finish")
    print(post_server(state, "tell me a story"))
    print("post 3 finish")
    print(post_server(state, "can you help me to solve a question？"))
    print("post 4 finish")
    question = "The capital structure of Ricketti Enterprises, Inc., consists of 15 million shares of common stock and 1 million warrants. Each warrant gives its owner the right to purchase 1 share of common stock for an exercise price of $19. The current stock price is $25, and each warrant is worth $7. All warrant holders choose to exercise their warrants today, this will lead to the issuance of new shares, what is the new stock price?"
    print(post_server(state, question))
    print("post 5 finish")
