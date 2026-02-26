from typing import Literal
import pickle

import torch
from datasets import Dataset, DatasetDict
from alignment.data import is_openai_format
from peft.utils import ModulesToSaveWrapper
from peft.tuners.tuners_utils import BaseTunerLayer

MISTRAL_CHAT_TEMPLATE = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

def load_dataset(
    data_path: str,
    author_id: int = 0,
    num_instances: int = 7,
    split: Literal["train", "test", "val"] = "train",
) -> DatasetDict:
    raw_dict = {}
    for dataset_key, path in [(split, data_path)]:
        
        prefs = {
            "prompt": [],
            "chosen": [],
        }
        
        with open(path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        
        spec_dataset = data[int(author_id)]    

        if num_instances:
            spec_dataset = spec_dataset[:int(num_instances)]
        
        for item in spec_dataset:
            prefs["prompt"].append(item["prompt"])

            prefs["chosen"].append([
                {
                    "content": item["prompt"],
                    "role": "user"
                },
                {
                    "content": item["output"].strip(),
                    "role": "assistant"
                }
            ])
            
        raw_dict[dataset_key] = Dataset.from_dict(prefs)

    return DatasetDict(raw_dict)

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "ditto"]
):
    
    tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
        
    if task in ["sft", "generation"]:
        
        messages = example["chosen"]
            
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
        
    elif task == "ditto":
        if not is_openai_format(example["chosen"]):
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
            )
        
        if "prompt" in example and is_openai_format(example["prompt"]):
            prompt_messages = example["prompt"]
            chosen_messages = example["chosen"]
        else:
            prompt_messages = example["chosen"][:-1]
            chosen_messages = example["chosen"][-1:]

        example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        if example["text_chosen"].startswith(tokenizer.bos_token):
            example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):]

    return example

def copy_adapter_weights(src_adapter_name, tgt_adapter_name, model):

    lora_modules = [module for module in model.modules() if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper))]

    with torch.no_grad():
        for model_module in lora_modules:
    
            if src_adapter_name in model_module.lora_A.keys():
                model_module.lora_A[tgt_adapter_name].load_state_dict(model_module.lora_A[src_adapter_name].state_dict())
                model_module.lora_B[tgt_adapter_name].load_state_dict(model_module.lora_B[src_adapter_name].state_dict())
    
            if src_adapter_name in model_module.lora_embedding_A.keys():
                model_module.lora_embedding_A[tgt_adapter_name].load_state_dict(model_module.lora_embedding_A[src_adapter_name].state_dict())
                model_module.lora_embedding_B[tgt_adapter_name].load_state_dict(model_module.lora_embedding_B[src_adapter_name].state_dict())
