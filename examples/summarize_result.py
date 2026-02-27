import os
import sys
import json
import logging
import argparse
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

from google import genai
import openai

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "outputs" / "evaluations"

def determine_provider(model_name: str) -> str:
    if "gpt" in model_name:
        return "openai"
    return "gemini"

def retrieve_results(identifier: str, provider: str = "gemini") -> list[dict[str, Any]]:
    if provider == "openai":
        
        client = openai.Client()
        batch_job = client.batches.retrieve(identifier)

        if batch_job.status == 'completed':
            if batch_job.output_file_id:
                result_file_id = batch_job.output_file_id
                file_response = client.files.content(result_file_id)
                file_content_str = file_response.read().decode('utf-8')
                
                json_objects = []
                for line in file_content_str.splitlines():
                    json_objects.append(json.loads(line))
                
                with open(OUTPUT_PATH / "batch_results.json", 'w', encoding='utf-8') as f:
                    json.dump(json_objects, f, indent=4) 
                
                return json_objects
        else:
            print(f"Job did not succeed. Final state: {batch_job.status}")
            if hasattr(batch_job, 'errors') and batch_job.errors:
                print(f"Error: {batch_job.errors}")
            sys.exit(0)
    else:  # gemini        
        client = genai.Client()
        batch_job = client.batches.get(name=identifier)

        if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
            if batch_job.dest and batch_job.dest.file_name:
                result_file_name = batch_job.dest.file_name
                file_content_bytes = client.files.download(file=result_file_name)
                file_content_str = file_content_bytes.decode('utf-8')
                
                json_objects = []
                for line in file_content_str.splitlines():
                    json_objects.append(json.loads(line))
                
                with open(OUTPUT_PATH / "batch_results.json", 'w', encoding='utf-8') as f:
                    json.dump(json_objects, f, indent=4) 
                
                return json_objects
        else:
            print(f"Job did not succeed. Final state: {batch_job.state.name}")
            if batch_job.error:
                print(f"Error: {batch_job.error}")
            sys.exit(0)

def process_leaderboard(matches: list[dict[str, Any]], provider: str = "gemini") -> dict[str, dict[str, list[int, int]]]:
    categories = {}

    for match in matches:
        if provider == "openai":
            custom_id = match["custom_id"]
            parts = custom_id.split("__")
            data_authid = parts[0]
            matchup = parts[1]
            winner = json.loads(match["response"]["body"]["choices"][0]["message"]["content"])["answer"]
        else:  # gemini
            key = match["key"]
            data_authid, matchup = key.split("__")
            winner = json.loads(match["response"]["candidates"][0]["content"]["parts"][0]["text"])["answer"]
        
        if data_authid not in categories.keys():
            categories[data_authid] = {}

        model_a, model_b = matchup.split("_VS_")

        if model_a not in categories[data_authid].keys():
            categories[data_authid][model_a] = [0, 0]

        if model_b not in categories[data_authid].keys():
            categories[data_authid][model_b] = [0, 0]

        categories[data_authid][model_a][winner != "A"] += 1
        categories[data_authid][model_b][winner == "A"] += 1

    with open(OUTPUT_PATH / "match_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(categories, f, indent=4)

    return categories

def calculate_winrate(statistics: dict[str, dict[str, list[int, int]]]) -> dict[str, dict[str, float]]:
    percentage = statistics.copy()
    for category, stats in statistics.items():
        for model, stat in stats.items():
            percentage[category][model] = stat[0] / (stat[0] + stat[1])
    
    with open(OUTPUT_PATH / "winrate.json", 'w', encoding='utf-8') as f:
        json.dump(percentage, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini"], help="LLM provider used")
    parser.add_argument("--batch_identifier", type=str, required=True, help="Batch job identifier to retrieve results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    
    provider = args.provider.lower()

    if provider == "openai":
        os.environ["api_key"] = os.environ.get("OPENAI_API_KEY", "")
    elif provider == "gemini":
        os.environ["api_key"] = os.environ.get("GEMINI_API_KEY", "")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    identifier = args.batch_identifier

    matches = retrieve_results(identifier, provider)
    statistics = process_leaderboard(matches, provider)
    calculate_winrate(statistics)


if __name__ == "__main__":
    main()