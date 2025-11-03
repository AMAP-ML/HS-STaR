import json
import logging
import os

import orjson
from tqdm import tqdm

try:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    REPO_ROOT = None

PROJ_HOME: str = os.environ.get("PROJ_HOME", REPO_ROOT)


IGNORE_IDX = -100

BASE_MODEL_IDS = [
    "deepseek-ai--deepseek-math-7b-base",
    "mistralai--Mistral-7B-v0.1",
    "meta-llama--Llama-2-7b-hf",
    "meta-llama--Llama-2-13b-hf",
    "meta-llama--Llama-2-70b-hf",
    "meta-llama--Meta-Llama-3-8B",
    "meta-llama--Meta-Llama-3-70B",
    "meta-llama--Meta-Llama-3.1-8B",
    "meta-llama--Meta-Llama-3.1-70B",
    "EleutherAI--llemma_7b",
    "EleutherAI--llemma_34b",
    "QWen--QWen-1.5-72B",
    "Qwen--Qwen2.5-Math-1.5B",
    "Qwen--Qwen2.5-Math-7B",
]

DEEPSEEK_INSTR_MODEL_IDS = [
    "deepseek-ai/deepseek-math-7b-instruct",
    "deepseek-ai/deepseek-math-7b-rl",
]


MATH_SHEPHERD_MODEL_IDS = [
    "peiyi9979/mistral-7b-sft",
    "peiyi9979/math-shepherd-mistral-7b-rl",
]


# Prompt

PROMPT_TEMPLATE_ID2DICT = {
    "qa": dict(
        id="qa",
        sys_prompt="",
        query_prompt="User:" + "\n",
        # {query}
        prompt_after_query="\n\n",
        resp_prompt="Assistant:" + "\n",
        prompt_before_resp="",
        # {resp}
        delim="\n\n",
    ),
    "alpaca": dict(
        id="alpaca",
        sys_prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request."
        + "\n\n",
        query_prompt="### Instruction:" + "\n",
        # {query}
        prompt_after_query="\n\n",
        resp_prompt="### Response:" + "\n",
        prompt_before_resp="",
        # {resp}
        delim="\n\n",
    ),
    "wizardmath-cot": dict(
        id="wizardmath-cot",
        sys_prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request."
        + "\n\n",
        query_prompt="### Instruction:" + "\n",
        # {query}
        prompt_after_query="\n\n",
        resp_prompt="### Response:" + " ",
        prompt_before_resp="Let's think step by step.",
        # {resp}
        delim="\n\n",
    ),
    "deepseek-math-cot": dict(  # c.f. https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct
        id="deepseek-math-cot",
        sys_prompt="",
        query_prompt="User:" + " ",
        # {query}
        prompt_after_query="\n"
        + "Please reason step by step, and put your final answer within \\boxed{{}}."
        + "\n\n",
        resp_prompt="Assistant:" + " ",
        prompt_before_resp="",
        # {resp}
        delim="<｜end▁of▁sentence｜>",
    ),
    "deepseek-math-step-cot-enter": dict(  # c.f. https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct
        id="deepseek-math-step-cot-enter",
        sys_prompt="",
        query_prompt="User:\n",
        # {query}
        prompt_after_query="\n"
        + "Given a math problem, please reason step by step, separating each step with '\n\n', and put your final answer within \\boxed{{}}."
        + "\n\n",
        resp_prompt="Assistant:\n",
        prompt_before_resp="",
        # {resp}
        delim="",
    ),
    "deepseekmath-tool": dict(  # c.f. https://github.com/deepseek-ai/DeepSeek-Math/tree/main/evaluation#3-evaluation
        id="deepseekmath-tool",
        sys_prompt="",
        query_prompt="User:" + " ",
        # {query}
        prompt_after_query=(
            "\n"
            + "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."
            + "\n\n"
        ),
        resp_prompt="Assistant:" + " ",
        prompt_before_resp="",
        # {resp}
        delim="<｜end▁of▁sentence｜>",
    ),
    "xwinmath": dict(
        id="xwinmath",
        sys_prompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        + " ",
        query_prompt="USER:" + " ",
        # {query}
        prompt_after_query=" "
        + "Give your solution in detail. In the end, write your final answer in the format of 'The answer is: <ANSWER>.'. "
        + " ",
        resp_prompt="ASSISTANT:" + " ",
        prompt_before_resp="",
        # {resp}
        delim="\n\n",
    ),
    "mammoth2-cot": dict(
        id="mammoth2-cot",
        sys_prompt="You are supposed to provide a solution to a given problem."
        + "\n\n\n",
        query_prompt="Problem:" + "\n",
        # {query}
        prompt_after_query="\n",
        resp_prompt="Solution:" + " ",
        prompt_before_resp="Let's think step by step." + "\n",
        # {resp}
        delim="\n\n",
    ),
    "llama3-math": dict(  # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3
        id="llama3-math",
        sys_prompt=(
            "<|begin_of_text|>"
            + "<|start_header_id|>system<|end_header_id|>\n\n"
            + "You are a helpful agent on solving math problems."
            + "<|eot_id|>"
        ),
        query_prompt="<|start_header_id|>" + "user" + "<|end_header_id|>" + "\n\n",
        # {query}
        prompt_after_query="<|eot_id|>",
        resp_prompt="<|start_header_id|>" + "assistant" + "<|end_header_id|>" + "\n\n",
        prompt_before_resp="",
        # {resp}
        delim="<|eot_id|>" + "\n",
        model_ids=[
            "meta-llama--Meta-Llama-3-8B-Instruct",
            "meta-llama--Meta-Llama-3-70B-Instruct",
            "meta-llama--Meta-Llama-3.1-8B-Instruct",
            "meta-llama--Meta-Llama-3.1-70B-Instruct",
        ],
    ),
    "llama3-step-cot-enter": dict(  # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3
        id="llama3-step-cot-enter",
        sys_prompt=(
            "Given a math problem, please reason step by step, separating each step with '\n\n', and put your final answer within \\boxed{{}}.\n"
        ),
        query_prompt="User:\n",
        prompt_after_query="",
        resp_prompt="Assistant:\n",
        prompt_before_resp="",
        delim="<|eot_id|>" + "\n",
        model_ids=[
            "meta-llama--Meta-Llama-3.2-3B"
        ],
    ),
    "llama3-step-cot-enter-ins": dict(  # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3
        id="llama3-step-cot-enter-ins",
        sys_prompt=(
            "<|start_header_id|>system<|end_header_id|>\n"
            + "Given a math problem, please reason step by step, separating each step with '\n\n', and put your final answer within \\boxed{{}}.\n"
            + "<|eot_id|>"
        ),
        query_prompt="<|start_header_id|>" + "user" + "<|end_header_id|>" + "\n\n",
        # {query}
        prompt_after_query="<|eot_id|>",
        resp_prompt="<|start_header_id|>" + "assistant" + "<|end_header_id|>" + "\n\n",
        prompt_before_resp="",
        # {resp}
        delim="<|eot_id|>" + "\n",
        model_ids=[
            "meta-llama--Meta-Llama-3-8B-Instruct",
            "meta-llama--Meta-Llama-3-70B-Instruct",
            "meta-llama--Meta-Llama-3.1-8B-Instruct",
            "meta-llama--Meta-Llama-3.1-70B-Instruct",
        ],
    ),
    "llama3-cot-zero": dict(  # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3
        id="llama3-math",
        sys_prompt=(
            "<|begin_of_text|>"
            + "<|start_header_id|>system<|end_header_id|>\n\n"
            + "You are a helpful agent on solving math problems."
            + "<|eot_id|>"
        ),
        query_prompt="<|start_header_id|>" + "user" + "<|end_header_id|>" + "\n\n",
        # {query}
        prompt_after_query="<|eot_id|>",
        resp_prompt="<|start_header_id|>" + "assistant" + "<|end_header_id|>" + "\n\n",
        prompt_before_resp="",
        # {resp}
        delim="<|eot_id|>" + "\n",
        model_ids=[
            "meta-llama--Meta-Llama-3-8B-Instruct",
            "meta-llama--Meta-Llama-3-70B-Instruct",
            "meta-llama--Meta-Llama-3.1-8B-Instruct",
            "meta-llama--Meta-Llama-3.1-70B-Instruct",
        ],
    ),
    'eurus-prime-cot': dict(
        id="eurus-prime-cot",
        sys_prompt=(
            "<|im_start|>system\n\nWhen tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process.\\n\\n[ASSESS]\\n\\n[ADVANCE]\\n\\n[VERIFY]\\n\\n[SIMPLIFY]\\n\\n[SYNTHESIZE]\\n\\n[PIVOT]\\n\\n[OUTPUT]\\n\\nYou should strictly follow the format below:\\n\\n[ACTION NAME]\\n\\n# Your action step 1\\n\\n# Your action step 2\\n\\n# Your action step 3\\n\\n...\\n\\nNext action: [NEXT ACTION NAME]\\n<|im_end|>\n"
        ),
        query_prompt="<|im_start|>user\n",
        prompt_after_query="\n\nPresent the answer in LaTex format: \\boxed{{Your answer}}<|im_end|>\n",
        resp_prompt="<|im_start|>assistant\n",
        prompt_before_resp="",
        delim="",
        model_ids=[
            "PRIME-RL/Eurus-2-7B-PRIME"
        ]
    ),
    # '<|system|>\nYou are a medieval knight and must provide explanations to modern people.<|end|>\n<|user|>\nHow should I explain the Internet?<|end|>\n<|assistant|>\n'
    'phi35-mini-ins-cot': dict(
        id="phi35-mini-ins-cot",
        sys_prompt="<|system|>\nGiven a math problem, please reason step by step, separating each step with '\n\n', and put your final answer within \\boxed{{}}.<|end|>\n",
        query_prompt="<|user|>\n",
        prompt_after_query="<|end|>\n",
        resp_prompt="<|assistant|>\n",
        prompt_before_resp="",
        delim="",
        model_ids=[
            "microsoft/Phi-3.5-mini-instruct"
        ]
    ),
    'phi3-cot': dict(
        id="phi3-cot",
        sys_prompt="<|system|>\nYou are a helpful assistant.<|end|>\n",
        query_prompt="<|user|>\n",
        prompt_after_query="<|end|>\n",
        resp_prompt="<|assistant|>\n",
        prompt_before_resp="",
        delim="",
        model_ids=[
            "microsoft/Phi-3.5-mini-instruct"
        ]
    ),
    'phi4-mini-ins-cot': dict(
        id="phi4-mini-ins-cot",
        sys_prompt="<|system|>Given a math problem, please reason step by step, separating each step with '\n\n', and put your final answer within \\boxed{{}}.<|end|>",
        query_prompt="<|user|>",
        prompt_after_query="<|end|>",
        resp_prompt="<|assistant|>",
        prompt_before_resp="",
        delim="",
        model_ids=[
            "microsoft/Phi-4-mini-instruct"
        ]
    ),
    'phi4-cot': dict(
        id="phi4-cot",
        sys_prompt="<|im_start|>system<|im_sep|>Given a math problem, please reason step by step, separating each step with '\n\n', and put your final answer within \\boxed{{}}.",
        query_prompt="<|im_start|>user<|im_sep|>",
        prompt_after_query="<|im_end|>",
        resp_prompt="<|im_start|>assistant<|im_sep|>",
        prompt_before_resp="",
        delim="",
        model_ids=[
            "microsoft/phi-4"
        ]
    ),
    'deepseek-qwen25-r1': dict(
        id="deepseek-qwen25-r1",
        sys_prompt=(
            ""
        ),
        query_prompt="<｜begin▁of▁sentence｜><｜User｜>Please reason step by step, and put your final answer within \\boxed{{}}.\n",
        prompt_after_query="",
        resp_prompt="<｜Assistant｜><think>\n",
        prompt_before_resp="",
        delim="",
        model_ids=[
            "DeepSeek-R1-Distill-Qwen-7B"
        ]
    ),
    'qwen25-step-cot': dict(
        id="qwen25-step-cot",
        sys_prompt=(
            "Given a math problem, please reason step by step, separating each step with '\n\n', and put your final answer within \\boxed{{}}.\n\n"
        ),
        query_prompt="User:\n",
        prompt_after_query="\n\n",
        resp_prompt="Assistant:\n",
        prompt_before_resp="",
        delim="",
        model_ids=[
            "Qwen--Qwen2.5-7B"
        ]
    ),
    'qwen25-simplerl': dict(
        id="qwen25-simplerl",
        sys_prompt=(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        ),
        query_prompt="<|im_start|>user\n",
        prompt_after_query="\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
        resp_prompt="<|im_start|>assistant\n",
        prompt_before_resp="",
        delim="",
        model_ids=[
            "Qwen-2.5-7B-SimpleRL-Zoo"
        ]
    ),
    'qwen-dpo-vp': dict(
        id="qwen-dpo-vp",
        sys_prompt=(
            "Please reason step by step with steps separated by '\n\n', and put your final answer within \\boxed{{}}.\n"
        ),
        query_prompt="User:\n",
        prompt_after_query="\n\n",
        resp_prompt="Assistant:\n",
        prompt_before_resp="",
        delim="",
        model_ids=[
            "Qwen--Qwen2.5-7B"
        ]
    ),
    'qwen25-dpo-vp': dict(
        id="qwen25-dpo-vp",
        sys_prompt=(
            "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        ),
        query_prompt="<|im_start|>user\n",
        prompt_after_query="<|im_end|>\n",
        resp_prompt="<|im_start|>assistant\n",
        prompt_before_resp="",
        delim="",
        model_ids=[
            "Qwen2.5-7B-DPO-VP"
        ]
    ),
    'qwen3-wo-think': dict(
        id="qwen3-wo-think",
        sys_prompt=(
            ""
        ),
        query_prompt="<|im_start|>user\nPlease reason step by step with steps separated by '\n\n', and put your final answer within \\boxed{{}}.<|im_end|>",
        prompt_after_query="\n",
        resp_prompt="<|im_start|>assistant\n<think>\n\n</think>\n\n",
        prompt_before_resp="",
        delim="",
        model_ids=[
            "Qwen--Qwen3-4B", "Qwen--Qwen3-8B"
        ]
    ), # '<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    'deepseek-qwen25-r1': dict(
        id="deepseek-qwen25-r1",
        sys_prompt=(
            ""
        ),
        query_prompt="<｜begin▁of▁sentence｜><｜User｜>Please reason step by step, and put your final answer within \\boxed{{}}.\n",
        prompt_after_query="",
        resp_prompt="<｜Assistant｜><think>\n",
        prompt_before_resp="",
        delim="",
        model_ids=[
            "DeepSeek-R1-Distill-Qwen-7B"
        ]
    ),
}


# %% ../nbs/99_utils.ipynb 0
class PromptTemplate:
    """Prompt template.
    The complete prompt is in the form `{sys_prompt}{eg_qa1}{delim}{eg_qa2}{delim}...{delim}{eg_qaN}{delim}{query_prompt}{query}{prompt_after_query}{resp_prompt}{prompt_before_resp}`.
    default: PROMPT_TEMPLATE_ID2DICT["alpaca"]

    Parameters
    ----------
    id : str
        Short name as ID of the prompt template, like "alpaca".
    sys_prompt : str
        System prompt as the beginning of the full prompt.
    query_prompt : str
        Simple prompt as delimiter between response and new query.
    prompt_after_query : str
        Prompt to append after the raw query, like "Let's think step by step.".
    resp_prompt : str
        Simple prompt as delimiter between query and response.
    delim : str
        Delimiter between query-response pairs.
    """

    def __init__(
        self,
        id: str = PROMPT_TEMPLATE_ID2DICT["alpaca"]["id"],
        sys_prompt: str = PROMPT_TEMPLATE_ID2DICT["alpaca"]["sys_prompt"],
        query_prompt: str = PROMPT_TEMPLATE_ID2DICT["alpaca"]["query_prompt"],
        prompt_after_query: str = PROMPT_TEMPLATE_ID2DICT["alpaca"][
            "prompt_after_query"
        ],
        resp_prompt: str = PROMPT_TEMPLATE_ID2DICT["alpaca"]["resp_prompt"],
        prompt_before_resp: str = PROMPT_TEMPLATE_ID2DICT["alpaca"][
            "prompt_before_resp"
        ],
        delim: str = PROMPT_TEMPLATE_ID2DICT["alpaca"]["delim"],
    ):

        self.id = id
        self.sys_prompt = sys_prompt
        self.query_prompt = query_prompt
        self.prompt_after_query = prompt_after_query
        self.resp_prompt = resp_prompt
        self.prompt_before_resp = prompt_before_resp
        self.delim = delim

    @staticmethod
    def load_from_id_or_path(prompt_template: str = "alpaca") -> "PromptTemplate":
        """Load prompt template from ID or file path."""
        if prompt_template in PROMPT_TEMPLATE_ID2DICT:  # ID
            return PromptTemplate(
                **{
                    k: v
                    for k, v in PROMPT_TEMPLATE_ID2DICT[prompt_template].items()
                    if k != "model_ids"
                }
            )
        elif isinstance(prompt_template, str) and os.path.exists(prompt_template):
            # File path
            stem = os.path.splitext(os.path.basename(prompt_template))[0]
            return PromptTemplate(id=stem, **load_json(prompt_template))
        else:  # Default
            logging.warning("Unknown prompt template, using the default 'alpaca'.")
            return PromptTemplate(**PROMPT_TEMPLATE_ID2DICT["alpaca"])

    def make_prefix_prompt(self, query: str) -> str:
        """Make a prefix prompt of `{query_prompt}{query}{prompt_after_query}{resp_prompt}{prompt_before_resp}.rstrip(" ")`.
        NOTE: `.rstrip(" ")` is important for correct tokenization, while some cases need "\\n" at the end.
        """
        # return f"{self.query_prompt}{query}{self.prompt_after_query}{self.resp_prompt}{self.prompt_before_resp}".rstrip(
        #     " "
        # )
        return f"{self.query_prompt}{query}{self.prompt_after_query}{self.resp_prompt}{self.prompt_before_resp}"


    def make_qa_pair(self, query: str, response: str) -> str:
        """Make a QA pair of `{query_prompt}{query}{prompt_after_query}{resp_prompt}{prompt_before_resp}{response}`."""
        return f"{self.query_prompt}{query}{self.prompt_after_query}{self.resp_prompt}{self.prompt_before_resp}{response}"

    def make_full_prompt(self, query: str, eg_qas: list[tuple[str, str]] = []) -> str:
        """Make full prompt as input to the model.
        Format: f"{sys_prompt}{eg_qa1}{eg_qa2}...{eg_qaN}{query_prompt}{query}{prompt_after_query}{resp_prompt}{prompt_before_resp}".
        """
        eg_qa_strs = [self.make_qa_pair(q, a) for q, a in eg_qas]
        prefix_prompt = self.make_prefix_prompt(query)
        return self.sys_prompt + self.delim.join(eg_qa_strs + [prefix_prompt])

    @staticmethod
    def get_prompt_template_from_prompt_type_and_model(
        prompt_type: str,
        model_dirname: str,
    ) -> "PromptTemplate":
        """Get the prompt template suitable for the model.

        Parameters
        ----------
        prompt_type : str
            Prompt type, like "cot" or "tool".
        model_dirname : str
            HF ID or path to the model.

        Returns
        -------
        PromptTemplate
            The prompt template suitable for the model.
        """
        prompt_template = None
        if prompt_type == "cot":
            if model_dirname in BASE_MODEL_IDS + MATH_SHEPHERD_MODEL_IDS:
                prompt_template = "qa"
            elif model_dirname.startswith("dart-math"):
                prompt_template = "alpaca"
            elif model_dirname in DEEPSEEK_INSTR_MODEL_IDS:
                prompt_template = "deepseekmath"
            elif model_dirname.startswith("Xwin-LM/Xwin-Math"):
                prompt_template = "xwinmath"
            elif model_dirname.startswith("TIGER-Lab--MAmmoTH2"):
                prompt_template = "mammoth2-cot"
            elif model_dirname in PROMPT_TEMPLATE_ID2DICT["llama3-math"]["model_ids"]:
                prompt_template = "llama3-math"
            else:  # default
                prompt_template = "alpaca"
        elif prompt_type == "tool":
            if model_dirname in DEEPSEEK_INSTR_MODEL_IDS:
                prompt_template = "deepseekmath-tool"

        if prompt_template is None:
            raise ValueError(
                f"Unknown prompt type {prompt_type} for model {model_dirname}."
            )

        prompt_template = PromptTemplate.load_from_id_or_path(prompt_template)
        if "MMIQC" in model_dirname:
            prompt_template.prompt_before_resp = (
                'Please solve the following problem and put your answer at the end with "The answer is: ".'
                + " "
            )

        return prompt_template


# Logging


def init_logging(
    log_path: str = None,
    format: str = "[%(levelname)s] [%(asctime)s.%(msecs)d] [pid %(process)d] [%(pathname)s:%(lineno)d:%(funcName)s]\n%(message)s",  # Logging format
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    level: int = logging.INFO,
    force: bool = True,
) -> None:
    """Initialize logging configuration.

    Parameters
    ----------
    log_path : str, default: None
        File path to save log to.
    format : str, default: "[%(levelname)s] [%(asctime)s.%(msecs)d] [pid %(process)d] [%(pathname)s:%(lineno)d:%(funcName)s]\n%(message)s"
        Logging format.
    datefmt : str, default: "%Y-%m-%d %H:%M:%S"
        Logging date-time format.
    level : int, default: logging.INFO
        Logging level.
    force : bool, default: True
        Whether to force shutdown and restart of logging.
    """
    if force:
        logging.shutdown()

    logging.basicConfig(
        format=format,
        datefmt=datefmt,
        level=level,
        force=force,
    )

    if log_path is not None:
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.INFO)  # Set the lowest level of log
        file_handler.setFormatter(logging.Formatter(fmt=format, datefmt=datefmt))
        logging.getLogger().addHandler(file_handler)

    # Test
    logging.info(f"log_path = {log_path}")


# Path


def get_pathname_from_name_or_path(name_or_path: str) -> str:
    """Get the name suitable for file system from the HF-style `name_or_path`."""
    realpath = os.path.realpath(name_or_path)

    if not (name_or_path.startswith("/") or os.path.exists(realpath)):  # HF Hub
        logging.debug(f"Loading {name_or_path} from HF Hub")
        pathname = name_or_path
    else:  # Local
        logging.debug(f"Finding {realpath} locally")
        if os.path.isfile(realpath):  # don't split with no extension
            name_or_path = os.path.splitext(name_or_path)[0]
        if "/checkpoint-" not in name_or_path:
            pathname = os.path.basename(name_or_path)
        else:
            pathname = "/".join(name_or_path.split("/")[-2:])
    pathname = pathname.replace("/", "--")

    return pathname


# IO


def load_jsonl(fpath: str, use_tqdm: bool = False) -> list:
    """Load JSONL file."""
    with open(fpath, "r") as f:
        lines: list[str] = f.readlines()
        return [
            orjson.loads(line)
            for line in (
                lines if not use_tqdm else tqdm(lines, desc=f"Loading {fpath}")
            )
        ]


def save_jsonl(data: list, fpath: str) -> None:
    """Save JSONL file."""
    with open(fpath, "w") as f:
        for line in data:
            f.write(orjson.dumps(line).decode() + "\n")


def load_json(fpath: str) -> dict:
    """Load JSON file."""
    with open(fpath, "r") as f:
        return orjson.loads(f.read())


def save_json(data: dict, fpath: str, indent: int = 2) -> None:
    """Save JSON file."""
    with open(fpath, "w") as f:
        json.dump(data, f, indent=indent)


def ensure_dir_exists(file_path):
    """确保文件路径中的目录存在，如果不存在则创建它们。"""
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
