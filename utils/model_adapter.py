"""Model adapter registration."""

import os
import sys
from typing import List
import warnings

if sys.version_info >= (3, 9):
    from functools import cache
else:
    from functools import lru_cache as cache

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    T5Tokenizer,
)

from utils.conversation import Conversation, get_conv_template

# Check an environment variable to check if we should be sharing Peft model
# weights.  When false we treat all Peft models as separate.
peft_share_base_weights = (
    os.environ.get("PEFT_SHARE_BASE_WEIGHTS", "false").lower() == "true"
)


class BaseModelAdapter:
    """The base and the default model adapter."""

    use_fast_tokenizer = True

    def match(self, model_path: str):
        return True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=self.use_fast_tokenizer,
                revision=revision,
                trust_remote_code=True,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, revision=revision, trust_remote_code=True
            )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **from_pretrained_kwargs,
            )
        except NameError:
            model = AutoModel.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **from_pretrained_kwargs,
            )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")


# A global registry for all model adapters
# TODO (lmzheng): make it a priority queue.
model_adapters: List[BaseModelAdapter] = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


@cache
def get_model_adapter(model_path: str) -> BaseModelAdapter:
    """Get a model adapter for a model_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_path))

    # Try the basename of model_path at first
    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModelAdapter:
            return adapter

    # Then try the full path
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter

    raise ValueError(f"No valid model adapter for {model_path}")


def raise_warning_for_incompatible_cpu_offloading_configuration(
    device: str, load_8bit: bool, cpu_offloading: bool
):
    if cpu_offloading:
        if not load_8bit:
            warnings.warn(
                "The cpu-offloading feature can only be used while also using 8-bit-quantization.\n"
                "Use '--load-8bit' to enable 8-bit-quantization\n"
                "Continuing without cpu-offloading enabled\n"
            )
            return False
        if not "linux" in sys.platform:
            warnings.warn(
                "CPU-offloading is only supported on linux-systems due to the limited compatability with the bitsandbytes-package\n"
                "Continuing without cpu-offloading enabled\n"
            )
            return False
        if device != "cuda":
            warnings.warn(
                "CPU-offloading is only enabled when using CUDA-devices\n"
                "Continuing without cpu-offloading enabled\n"
            )
            return False
    return cpu_offloading


def get_conversation_template(model_path: str) -> Conversation:
    """Get the default conversation template."""
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)


def add_model_args(parser):
    parser.add_argument(
        "--model-path",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face Hub model revision identifier",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu", "npu"],
        default="cuda",
        help="The device type",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per GPU for storing model weights. Use a string like '13Gib'",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--cpu-offloading",
        action="store_true",
        help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU",
    )
    parser.add_argument(
        "--gptq-ckpt",
        type=str,
        default=None,
        help="Used for GPTQ. The path to the local GPTQ checkpoint.",
    )
    parser.add_argument(
        "--gptq-wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="Used for GPTQ. #bits to use for quantization",
    )
    parser.add_argument(
        "--gptq-groupsize",
        type=int,
        default=-1,
        help="Used for GPTQ. Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--gptq-act-order",
        action="store_true",
        help="Used for GPTQ. Whether to apply the activation order GPTQ heuristic",
    )
    parser.add_argument(
        "--awq-ckpt",
        type=str,
        default=None,
        help="Used for AWQ. Load quantized model. The path to the local AWQ checkpoint.",
    )
    parser.add_argument(
        "--awq-wbits",
        type=int,
        default=16,
        choices=[4, 16],
        help="Used for AWQ. #bits to use for AWQ quantization",
    )
    parser.add_argument(
        "--awq-groupsize",
        type=int,
        default=-1,
        help="Used for AWQ. Groupsize to use for AWQ quantization; default uses full row.",
    )
    parser.add_argument(
        "--enable-exllama",
        action="store_true",
        help="Used for exllamabv2. Enable exllamaV2 inference framework.",
    )
    parser.add_argument(
        "--exllama-max-seq-len",
        type=int,
        default=4096,
        help="Used for exllamabv2. Max sequence length to use for exllamav2 framework; default 4096 sequence length.",
    )
    parser.add_argument(
        "--exllama-gpu-split",
        type=str,
        default=None,
        help="Used for exllamabv2. Comma-separated list of VRAM (in GB) to use per GPU. Example: 20,7,7",
    )
    parser.add_argument(
        "--exllama-cache-8bit",
        action="store_true",
        help="Used for exllamabv2. Use 8-bit cache to save VRAM.",
    )
    parser.add_argument(
        "--enable-xft",
        action="store_true",
        help="Used for xFasterTransformer Enable xFasterTransformer inference framework.",
    )
    parser.add_argument(
        "--xft-max-seq-len",
        type=int,
        default=4096,
        help="Used for xFasterTransformer. Max sequence length to use for xFasterTransformer framework; default 4096 sequence length.",
    )
    parser.add_argument(
        "--xft-dtype",
        type=str,
        choices=["fp16", "bf16", "int8", "bf16_fp16", "bf16_int8"],
        help="Override the default dtype. If not set, it will use bfloat16 for first token and float16 next tokens on CPU.",
        default=None,
    )


def remove_parent_directory_name(model_path):
    """Remove parent directory name."""
    if model_path[-1] == "/":
        model_path = model_path[:-1]
    return model_path.split("/")[-1]


peft_model_cache = {}


class OpenOrcaAdapter(BaseModelAdapter):
    """Model adapter for Open-Orca models which may use different prompt templates
    - (e.g. Open-Orca/OpenOrcaxOpenChat-Preview2-13B, Open-Orca/Mistral-7B-OpenOrca)
    - `OpenOrcaxOpenChat-Preview2-13B` uses their "OpenChat Llama2 V1" prompt template.
        - [Open-Orca/OpenOrcaxOpenChat-Preview2-13B #Prompt Template](https://huggingface.co/Open-Orca/OpenOrcaxOpenChat-Preview2-13B#prompt-template)
    - `Mistral-7B-OpenOrca` uses the [OpenAI's Chat Markup Language (ChatML)](https://github.com/openai/openai-python/blob/main/chatml.md)
        format, with <|im_start|> and <|im_end|> tokens added to support this.
        - [Open-Orca/Mistral-7B-OpenOrca #Prompt Template](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca#prompt-template)
    """

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return (
            "mistral-7b-openorca" in model_path.lower()
            or "openorca" in model_path.lower()
        )

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=self.use_fast_tokenizer, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        ).eval()
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "mistral-7b-openorca" in model_path.lower():
            return get_conv_template("mistral-7b-openorca")
        return get_conv_template("open-orca")


class DolphinAdapter(OpenOrcaAdapter):
    """Model adapter for ehartford/dolphin-2.2.1-mistral-7b"""

    def match(self, model_path: str):
        return "dolphin" in model_path.lower() and "mistral" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("dolphin-2.2.1-mistral-7b")


# Note: the registration order matters.
# The one registered earlier has a higher matching priority.
register_model_adapter(OpenOrcaAdapter)
register_model_adapter(DolphinAdapter)

# After all adapters, try the default base adapter.
register_model_adapter(BaseModelAdapter)
