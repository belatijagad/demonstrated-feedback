import os
import logging
import argparse
from pathlib import Path
from typing import Any

import torch
import pandas as pd
from peft import PeftModel
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from huggingface_hub import repo_exists

from scripts.utils import generate_model_outputs, seed_everything, MISTRAL_CHAT_TEMPLATE
from scripts.estimator import ESTIMATOR_MAP

logger = logging.getLogger(__name__)
logging.getLogger("transformers.pipelines").setLevel(logging.WARNING)


def process_dataset(
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    dataset_kwargs: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    
    author_id = dataset_kwargs["author_id"]
    n_train = dataset_kwargs["num_train_samples"]
    n_eval = dataset_kwargs["num_eval_samples"]
    target_eval_split = dataset_kwargs["eval_split"]

    # Load and filter train data for the specific author
    train_data = (
        dataset["train"]
        .filter(lambda x: x["author_id"] == author_id)
        .shuffle(seed=seed)
    )
    
    assert len(train_data) > 0, f"No rows found for author {author_id} in 'train' split."
    
    # Select training examples
    num_train_samples = min(n_train, len(train_data))
    examples = train_data.select(range(num_train_samples))

    # Load and filter eval data for the specific author (no shuffle)
    eval_data = (
        dataset[target_eval_split]
        .filter(lambda x: x["author_id"] == author_id)
    )
    
    assert len(eval_data) > 0, f"No rows found for author {author_id} in '{target_eval_split}' split."

    # Limit eval samples if specified
    if n_eval > 0:
        num_eval_samples = min(n_eval, len(eval_data))
        eval_data = eval_data.select(range(num_eval_samples))

    # Extract prompts from eval data for the author
    prompts: list[str] = [example["prompt"] for example in eval_data]

    return examples, prompts

def generate_examples(
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset, 
    dataset_kwargs: dict[str, Any],
    base_dir: str,
) -> None:
    assert dataset_kwargs["eval_split"] != "train", "Doesn't support `train` split currently."
    example_dataset = dataset[dataset_kwargs["eval_split"]].to_pandas()
    example_dataset = example_dataset.loc[example_dataset.author_id == dataset_kwargs["author_id"]]
    os.makedirs(base_dir+"/examples", exist_ok=True)
    example_dataset.to_csv(base_dir + "/examples/" + f"{dataset_kwargs['name']}_{dataset_kwargs['author_id']}.csv", index=False)

def generate_results(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    prompts: list[str],
    examples: Dataset,
    gen_kwargs: dict[str, Any],
    method_name: str,
    base_dir: str,
) -> None:
    responses_dict = {
        "prompt": [],
        "completion": [],
    }

    model_inputs = []
    csv_prompts = prompts 
    
    # Handle batch size and num_return_sequences
    gen_kwargs = gen_kwargs.copy()
    batch_size = gen_kwargs.pop("batch_size", 1)
    num_return_sequences = gen_kwargs.pop("num_return_sequences", 1)

    # 1. Format all prompts first
    for p in prompts:
        messages = []

        if method_name == "few-shot":
            for example in examples:
                messages.append({"role": "user", "content": example["prompt"]})
                messages.append({"role": "assistant", "content": example["chosen"]})

        messages.append({"role": "user", "content": p})

        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        model_inputs.append(formatted)

    all_completions = []

    # 2. Process in batches
    for i in range(0, len(model_inputs), batch_size):
        batch_prompts = model_inputs[i : i + batch_size]
        
        # New utility returns a LIST of tuples: [(prompt_ids, gen_ids, scores, logits), ...]
        batch_results = generate_model_outputs(
            prompts=batch_prompts,
            model=model,
            tokenizer=tokenizer,
            gen_kwargs=gen_kwargs,
            num_return_sequences=num_return_sequences,
        )
        
        # 3. Iterate through the results list (each result is a dict)
        for result in batch_results:
            gen_ids = result["gen_ids"]
            decoded_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_completions.append(decoded_text)
            
    # 4. Save to CSV
    for raw_prompt, completion in zip(csv_prompts, all_completions, strict=True):
        clean_prompt = raw_prompt.replace('\n', '\\n')
        responses_dict["prompt"].append(clean_prompt)
        
        clean_completion = completion.strip().replace('\n', '\\n')
        responses_dict["completion"].append(clean_completion)

    responses = pd.DataFrame.from_dict(responses_dict)
    os.makedirs(base_dir, exist_ok=True)
    responses.to_csv(f"{base_dir}/{method_name}.csv", index=False)
    
def main():
    parser = argparse.ArgumentParser(description="Generation script")
    parser.add_argument("-d", "--dataset_name_or_path", type=str, required=True, help="HuggingFace dataset name or path")
    parser.add_argument("-a", "--author_id", type=str, required=True, help="Author ID in the dataset")
    parser.add_argument("-m", "--model_name_or_path", type=str, required=True, help="Base model name or path")
    parser.add_argument("--model_name", type=str, default=None, help="Short model name (defaults to last part of model_name_or_path)")
    parser.add_argument("--dataset_name", type=str, default=None, help="Short dataset name (defaults to last part of dataset_name_or_path)")
    parser.add_argument("--checkpoint_base_dir", type=str, default="outputs/models", help="Base directory containing model checkpoints")
    parser.add_argument("--output_base_dir", type=str, default="outputs/generations", help="Base directory for generation output CSVs")
    parser.add_argument("--eval_split", type=str, default="val", help="Dataset split to use for evaluation")
    parser.add_argument("--num_train_samples", type=int, default=7, help="Number of train samples for few-shot")
    parser.add_argument("--num_eval_samples", type=int, default=3, help="Max eval samples (0 = use all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", default=True, help="Use sampling")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to generate per prompt")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    seed_everything(args.seed)

    # Derive short names: e.g. "belati/ccat50" -> "ccat50", "mistralai/Mistral-7B-Instruct-v0.2" -> "mistral-7b"
    model_name = args.model_name or args.model_name_or_path.split("/")[-1]
    dataset_name = args.dataset_name or args.dataset_name_or_path.split("/")[-1]

    # Directory structure matching config: {dataset_name}_{author_id}_{model_name}
    run_name = f"{dataset_name}_{args.author_id}_{model_name}"
    checkpoint_dir = Path(args.checkpoint_base_dir) / run_name
    output_dir = Path(args.output_base_dir) / run_name

    repo_id = f"belati/{model_name}_{dataset_name}_{args.author_id}"
    use_remote_repo = repo_exists(repo_id)

    logger.info(f"Source: {'HuggingFace Hub (' + repo_id + ')' if use_remote_repo else 'Local Checkpoints'}")

    dataset_config = {
        "name_or_path": args.dataset_name_or_path,
        "name": dataset_name,
        "author_id": args.author_id,
        "num_train_samples": args.num_train_samples,
        "num_eval_samples": args.num_eval_samples,
        "eval_split": args.eval_split,
    }

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "num_return_sequences": args.num_return_sequences,
        "batch_size": args.batch_size,
    }

    logger.info(f"Starting generation for model {args.model_name_or_path} on dataset {args.dataset_name_or_path}")

    dataset = load_dataset(dataset_config["name_or_path"])
    examples, prompts = process_dataset(dataset, dataset_config, args.seed)

    generate_examples(dataset, dataset_config, base_dir=str(output_dir.parent))

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    logger.info(f"Base tokenizer size: {len(tokenizer)}")

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "left"

    logger.info("Generating Zero-shot...")
    generate_results(model, tokenizer, prompts, examples, generation_config, method_name="zero-shot", base_dir=str(output_dir))

    logger.info("Generating Few-shot...")
    generate_results(model, tokenizer, prompts, examples, generation_config, method_name="few-shot", base_dir=str(output_dir))

    ref_model_path = repo_id if use_remote_repo else str(checkpoint_dir / "ref_model")

    logger.info(f"Loading SFT adapter from {ref_model_path}")
    model = PeftModel.from_pretrained(
        model=model,
        model_id=ref_model_path,
        subfolder="ref_model",
        adapter_name="ref_model",
        is_trainable=False,
    )

    generate_results(model, tokenizer, prompts, examples, generation_config, method_name="sft", base_dir=str(output_dir))
    logger.info("-> Finished generating SFT generations.")

    for name in ESTIMATOR_MAP.keys():
        adapter_name = f"{name}_policy_model"
        if use_remote_repo:
            adapter_path = repo_id
        else:
            adapter_path = str(checkpoint_dir / adapter_name)
            if not os.path.exists(adapter_path):
                logger.warning(f"Local adapter {adapter_path} not found. Skipping.")
                continue

        logger.info(f"Loading Adapter: {name} from {adapter_path}")

        try:
            model.load_adapter(adapter_path, adapter_name=adapter_name, subfolder=adapter_name)
            model.set_adapter(adapter_name)

            generate_results(model, tokenizer, prompts, examples, generation_config, method_name=name, base_dir=str(output_dir))
            logger.info(f"-> Finished generating DITTO {name} generations.")

            model.delete_adapter(adapter_name)

        except Exception as e:
            logger.error(f"Failed to load or generate for {name}: {e}")
    
if __name__ == "__main__":
    main()
