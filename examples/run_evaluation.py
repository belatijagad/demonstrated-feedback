import os
import csv
import json
import logging
import argparse
import textwrap
from typing import Any
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from itertools import permutations
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = ROOT / "outputs" / "generations"
EXAMPLES_DIR = GEN_DIR / "examples"
OUTPUT_PATH = ROOT / "outputs" / "evaluations"

class JudgeResult(BaseModel):
    answer: Literal["A", "B"] = Field(description="The option most similar to the HUMAN AUTHOR'S WRITING; either A or B")
    reasoning: str = Field(description="Brief explanation of why this option is more similar")

PROMPT_TEMPLATE_GEMINI = textwrap.dedent(
    """You are an impartial evaluator.
    Below is a sample of a human author"s writing and two options.

    ### HUMAN AUTHOR"S WRITING:
    {demo}

    ### OUTPUT A:
    {text_a}

    ### OUTPUT B:
    {text_b}

    ### Task
    Which option was written by the human author based on similarity to the HUMAN AUTHOR"S WRITING above?

    ALWAYS REMAIN IMPARTIAL WHEN EVALUATING OUTPUTS.
    """
)

PROMPT_TEMPLATE_OPENAI = textwrap.dedent(
    """You are an impartial evaluator.
    Below is a sample of a human author"s writing and two options.

    ### HUMAN AUTHOR"S WRITING:
    {demo}

    ### OUTPUT A:
    {text_a}

    ### OUTPUT B:
    {text_b}

    ### Task
    Which option was written by the human author based on similarity to the HUMAN AUTHOR"S WRITING above? Respond only with a JSON of the following format:
    {{
        "answer": "<The option most similar to the HUMAN AUTHOR'S WRITING; either A or B>"
    }}

    ALWAYS REMAIN IMPARTIAL WHEN EVALUATING OUTPUTS.
    """
)

def aggregate_responses() -> dict[str, Any]:
    data = {}

    def read_prompts(csv_path: Path):
        prompts, completions = [], []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            
            headers = reader.fieldnames or []
            if "completion" in headers:
                resp_key = "completion"
            elif "demo" in headers:
                resp_key = "demo"
            elif "chosen" in headers:
                resp_key = "chosen"
            else:
                resp_key = headers[1] if len(headers) > 1 else "completion"

            for row in reader:
                p_text = row.get("prompt", "").strip()
                c_text = row.get(resp_key, "").strip()

                if not p_text and not c_text:
                    continue
                
                prompts.append(p_text)
                completions.append(c_text)
                
        return prompts, completions

    for folder in sorted(GEN_DIR.iterdir()):
        if not folder.is_dir() or folder.name == "examples":
            continue
        
        # Folder format: {dataset}-{model}-{method}-key-{author_id}
        prefix, author_id = folder.name.rsplit("-key-", 1)
        dataset_name, model_info = prefix.split("-", 1)
        group_id = f"{dataset_name}_{author_id}"

        if group_id not in data:
            prompts, demos = read_prompts(EXAMPLES_DIR / f"{group_id}.csv")
            data[group_id] = {"prompt": prompts, "demo": demos, "generations": {}}

        entry = data[group_id]
        expected_len = len(entry["prompt"])

        for csv_file in sorted(folder.glob("*.csv")):
            gen_prompts, gen_completions = read_prompts(csv_file)
            actual_len = len(gen_prompts)

            if actual_len != expected_len:
                if actual_len > expected_len:
                    logger.warning(
                        f"MISMATCH FIXED: {csv_file.name} has {actual_len} prompts, "
                        f"truncating to expected {expected_len}."
                    )
                    gen_completions = gen_completions[:expected_len]
                else:
                    logger.error(
                        f"MISMATCH: {csv_file.name} has {actual_len} prompts, "
                        f"but expected {expected_len}. Skipping."
                    )
                    continue
            
            entry["generations"][f"{model_info}_{csv_file.stem}"] = gen_completions

    comparison_output = OUTPUT_PATH / "comparison.json"
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    with comparison_output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)

    logger.info("Finished aggregating data.")
    return data

def create_jsonl(data: dict[str, Any], args) -> None:
    requests = []
    provider = args.provider.lower()
    prompt_template = PROMPT_TEMPLATE_OPENAI if provider == "openai" else PROMPT_TEMPLATE_GEMINI

    for data_aid, content in data.items():
        prompts_list = content["prompt"]
        demos_list = content["demo"]
        generations_dict = content["generations"]
        
        model_names = list(generations_dict.keys())

        for model_name_a, model_name_b in permutations(model_names, 2):
            
            completions_a = generations_dict[model_name_a]
            completions_b = generations_dict[model_name_b]

            for i, (p_text, d_text, resp_a, resp_b) in enumerate(zip(prompts_list, demos_list, completions_a, completions_b, strict=True)):
                
                context_input = f"{p_text}\n{d_text}"
                option_a = f"{p_text}\n{resp_a}"
                option_b = f"{p_text}\n{resp_b}"

                if provider == "openai":
                    unique_id = f"{data_aid}__{model_name_a}_VS_{model_name_b}__i-{i}"
                    requests.append({
                        "custom_id": unique_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": args.model_name,
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt_template.format(demo=context_input, text_a=option_a, text_b=option_b)},
                            ]
                        },
                    })
                else:  # gemini
                    requests.append({
                        "key": f"{data_aid}__{model_name_a}_VS_{model_name_b}",
                        "request": {
                            "contents": [
                                {
                                    "parts": [
                                        {
                                            "text": prompt_template.format(demo=context_input, text_a=option_a, text_b=option_b)
                                        }
                                    ]
                                }
                            ],
                            "generationConfig": {
                                "response_mime_type": "application/json",
                                "response_json_schema": JudgeResult.model_json_schema(),
                            },
                        }
                    })

    comparison_output = OUTPUT_PATH / "batch-request.jsonl"
    comparison_output.parent.mkdir(parents=True, exist_ok=True)

    with comparison_output.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logger.info(f"Finished creating jsonl file with {len(requests)} requests.")

def create_batch_request(args) -> None:
    provider = args.provider.lower()
    
    if provider == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        client = openai.Client()
        uploaded_file = client.files.create(
            file=open(OUTPUT_PATH / "batch-request.jsonl", "rb"),
            purpose="batch",
        )
        
        file_batch_job = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        logger.info(f"=>> Created batch job: {file_batch_job.id}")
    else:  # gemini
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenAI library not available. Install with: pip install google-genai")
        
        client = genai.Client()
        uploaded_file = client.files.upload(
            file=OUTPUT_PATH / "batch-request.jsonl",
            config=types.UploadFileConfig(display_name="eval", mime_type="jsonl")
        )
        
        file_batch_job = client.batches.create(
            model=args.model_name,
            src=uploaded_file.name,
            config={
                "display_name": args.batch_display_name,
            },
        )
        
        logger.info(f"=>> Created batch job: {file_batch_job.name}")



def main():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini"], help="LLM provider to use")
    parser.add_argument("--model_name", type=str, default="gpt-4-0613", help="Model name for evaluation")
    parser.add_argument("--batch_display_name", type=str, default="evaluate", help="Display name for the batch job")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    
    provider = args.provider.lower()
    
    if provider == "openai":
        os.environ["api_key"] = os.environ.get("OPENAI_API_KEY", "")
        logger.info("=>> Using OpenAI provider")
    else:
        os.environ["api_key"] = os.environ.get("GEMINI_API_KEY", "")
        logger.info("=>> Using Gemini provider")
    
    logger.info("=>> Starting the evaluation process...")
    logger.info(f"Model for evaluation: {args.model_name}")
    data = aggregate_responses()
    create_jsonl(data, args)
    create_batch_request(args)


if __name__ == "__main__":
    main()
