#!/usr/bin/env python3

"""This python script is designed to run inference on a dataset using either the OpenAI or Anthropic API, depending on the model specified.
It sorts instances by length and continually writes the outputs to a specified file, so that the script can be stopped and restarted without losing progress.
"""

import json
import os
import time
import dotenv
import traceback
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import tiktoken
import openai
TOKENIZER_OVERRIDE = os.environ.get("TOKENIZER_OVERRIDE")  # "cl100k"
try:
    # Prefer OpenAI v1 client if available
    from openai import OpenAI as _OpenAIClient
except Exception:  # pragma: no cover
    _OpenAIClient = None
from anthropic import HUMAN_PROMPT, AI_PROMPT, Anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from datasets import load_dataset, load_from_disk
from swebench.inference.make_datasets.utils import extract_diff
from argparse import ArgumentParser
import logging
from dotenv import load_dotenv
from swebench.inference.langgraph_patch_flow import call_chat_via_langgraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

# Global default client for OpenAI v1 SDK when routing via OpenRouter
OPENAI_CLIENT = None

MODEL_LIMITS = {
    "claude-instant-1": 100_000,
    "claude-2": 100_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "gpt-3.5-turbo-16k-0613": 16_385,
    "gpt-3.5-turbo-0613": 4_097,
    "gpt-3.5-turbo-1106": 16_385,
    "gpt-4-32k-0613": 32_768,
    "gpt-4-0613": 8_192,
    "gpt-4-1106-preview": 128_000,
    "gpt-4-0125-preview": 128_000,
    "openai/gpt-5": 128_000, 
    "x-ai/grok-code-fast-1": 256_000
}

# The cost per token for each model input.
MODEL_COST_PER_INPUT = {
    "claude-instant-1": 0.00000163,
    "claude-2": 0.00001102,
    "claude-3-opus-20240229": 0.000015,
    "claude-3-sonnet-20240229": 0.000003,
    "claude-3-haiku-20240307": 0.00000025,
    "gpt-3.5-turbo-16k-0613": 0.0000015,
    "gpt-3.5-turbo-0613": 0.0000015,
    "gpt-3.5-turbo-1106": 0.000001,
    "gpt-35-turbo-0613": 0.0000015,
    "gpt-35-turbo": 0.0000015,  # probably still 0613
    "gpt-4-0613": 0.00003,
    "gpt-4-32k-0613": 0.00006,
    "gpt-4-32k": 0.00006,
    "gpt-4-1106-preview": 0.00001,
    "gpt-4-0125-preview": 0.00001,
    "openai/gpt-5": 0.00000125,
    "x-ai/grok-code-fast-1": 0.0000002,
}

# The cost per token for each model output.
MODEL_COST_PER_OUTPUT = {
    "claude-instant-1": 0.00000551,
    "claude-2": 0.00003268,
    "claude-3-opus-20240229": 0.000075,
    "claude-3-sonnet-20240229": 0.000015,
    "claude-3-haiku-20240307": 0.00000125,
    "gpt-3.5-turbo-16k-0613": 0.000002,
    "gpt-3.5-turbo-16k": 0.000002,
    "gpt-3.5-turbo-1106": 0.000002,
    "gpt-35-turbo-0613": 0.000002,
    "gpt-35-turbo": 0.000002,
    "gpt-4-0613": 0.00006,
    "gpt-4-32k-0613": 0.00012,
    "gpt-4-32k": 0.00012,
    "gpt-4-1106-preview": 0.00003,
    "gpt-4-0125-preview": 0.00003,
    "openai/gpt-5": 0.00001,
    "x-ai/grok-code-fast-1": 0.0000015
}

# used for azure
ENGINES = {
    "gpt-3.5-turbo-16k-0613": "gpt-35-turbo-16k",
    "gpt-4-0613": "gpt-4",
    "gpt-4-32k-0613": "gpt-4-32k",
}


# Fallback helpers for unknown model names (e.g., OpenRouter model ids)
def get_model_limit(model_name_or_path: str) -> int:
    """Return context length limit for a model, with a safe default for unknown models."""
    default_limit = int(os.environ.get("DEFAULT_MODEL_LIMIT", "128000"))
    return MODEL_LIMITS.get(model_name_or_path, default_limit)


def get_cost_per_input(model_name_or_path: str) -> float:
    """Return input token price; fall back to 0.0 if unknown."""
    return MODEL_COST_PER_INPUT.get(model_name_or_path, 0.0)


def get_cost_per_output(model_name_or_path: str) -> float:
    """Return output token price; fall back to 0.0 if unknown."""
    return MODEL_COST_PER_OUTPUT.get(model_name_or_path, 0.0)


def calc_cost(model_name, input_tokens, output_tokens):
    """
    Calculates the cost of a response from the openai API.

    Args:
    response (openai.ChatCompletion): The response from the API.

    Returns:
    float: The cost of the response.
    """
    cost = (
        get_cost_per_input(model_name) * input_tokens
        + get_cost_per_output(model_name) * output_tokens
    )
    logger.info(
        f"input_tokens={input_tokens}, output_tokens={output_tokens}, cost={cost:.2f}"
    )
    return cost


@retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
def call_chat(model_name_or_path, inputs, use_azure, temperature, top_p, **model_args):
    """
    Calls the openai API to generate completions for the given inputs.

    Args:
    model_name_or_path (str): The name or path of the model to use.
    inputs (str): The inputs to generate completions for.
    use_azure (bool): Whether to use the azure API.
    temperature (float): The temperature to use.
    top_p (float): The top_p to use.
    **model_args (dict): A dictionary of model arguments.
    """
    system_messages = inputs.split("\n", 1)[0]
    user_message = inputs.split("\n", 1)[1]
    try:
        # Choose client: prefer instantiated v1 client (e.g., OpenRouter),
        # otherwise fallback to module-level convenience API
        _client = OPENAI_CLIENT if OPENAI_CLIENT is not None else openai
        if use_azure:
            response = _client.chat.completions.create(
                engine=ENGINES[model_name_or_path] if use_azure else None,
                messages=[
                    {"role": "system", "content": system_messages},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                top_p=top_p,
                **model_args,
            )
        else:
            response = _client.chat.completions.create(
                model=model_name_or_path,
                messages=[
                    {"role": "system", "content": system_messages},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                top_p=top_p,
                **model_args,
            )
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = calc_cost(response.model, input_tokens, output_tokens)
        return response, cost
    except openai.BadRequestError as e:
        if e.code == "context_length_exceeded":
            print("Context length exceeded")
            return None
        raise e


def gpt_tokenize(string: str, encoding) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens


def claude_tokenize(string: str, api) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = api.count_tokens(string)
    return num_tokens


def choose_encoding(model_name_or_path: str):
    """Choose cl100k encoding for the model if TOKENIZER_OVERRIDE is set to cl100k."""
    if TOKENIZER_OVERRIDE == "cl100k":
        return tiktoken.get_encoding("cl100k_base")
    try:
        return tiktoken.encoding_for_model(model_name_or_path)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return tiktoken.get_encoding("p50k_base")


def openai_inference(
    test_dataset,
    model_name_or_path,
    output_file,
    model_args,
    existing_ids,
    max_cost,
):
    """
    Runs inference on a dataset using the openai API.

    Args:
    test_dataset (datasets.Dataset): The dataset to run inference on.
    model_name_or_path (str): The name or path of the model to use.
    output_file (str): The path to the output file.
    model_args (dict): A dictionary of model arguments.
    existing_ids (set): A set of ids that have already been processed.
    max_cost (float): The maximum cost to spend on inference.
    """
    # Try to get encoding for the model; fallback for unknown OpenRouter model ids
    try:
        encoding = tiktoken.choose_encoding(model_name_or_path)
    except Exception:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # As a last resort, use p50k_base which is widely available
            encoding = tiktoken.get_encoding("p50k_base")
    test_dataset_filtered = test_dataset.filter(
        lambda x: gpt_tokenize(x["text"], encoding) <= get_model_limit(model_name_or_path),
        desc="Filtering",
        load_from_cache_file=False,
    )
    # If over-filtered to zero (e.g., unknown model limit too small), keep unfiltered dataset with a warning
    if len(test_dataset_filtered) == 0 and len(test_dataset) > 0:
        logger.warning(
            f"After context filtering, 0 instances remain for model {model_name_or_path}. "
            f"Falling back to no-length-filtering for this run. Consider setting DEFAULT_MODEL_LIMIT to a larger value."
        )
    else:
        test_dataset = test_dataset_filtered
    
    # set the openrouter key (used for subset 10)
    or_key = os.environ.get("OPENROUTER_API_KEY", None)
    if or_key:
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        # Keep compatibility with both old and new SDKs
        openai.api_key = or_key
        try:
            openai.api_base = base_url  # old SDK
        except Exception:
            pass
        try:
            # new SDK global base_url
            openai.base_url = base_url  # type: ignore[attr-defined]
        except Exception:
            pass
        # Ensure env vars also reflect the desired base for any internal defaults
        os.environ["OPENAI_API_KEY"] = or_key
        os.environ["OPENAI_BASE_URL"] = base_url
        # Instantiate a dedicated v1 client if available and store globally
        global OPENAI_CLIENT
        if _OpenAIClient is not None:
            OPENAI_CLIENT = _OpenAIClient(api_key=or_key, base_url=base_url)
        print(f"Using OpenRouter key {'*' * max(0, len(or_key) - 5) + or_key[-5:]}")
    else:
        # elsewise use the openai key
        openai_key = os.environ.get("OPENAI_API_KEY", None)
        if openai_key is None:
            raise ValueError(
                "Must provide an api key. Expected in OPENROUTER_API_KEY or OPENAI_API_KEY environment variable."
            )
        openai.api_key = openai_key
        print(f"Using OpenAI key {'*' * max(0, len(openai_key) - 5) + openai_key[-5:]}")
    use_azure = model_args.pop("use_azure", False)
    if use_azure:
        openai.api_type = "azure"
        openai.api_base = "https://pnlpopenai3.openai.azure.com/"
        openai.api_version = "2023-05-15"

    # set temperature = 0, top_p = 0 for self-consistency
    temperature = model_args.pop("temperature", 0)
    top_p = model_args.pop("top_p", 0 if temperature == 0 else 1)
    print(f"Using temperature={temperature}, top_p={top_p}")
    use_langgraph = bool(model_args.pop("use_langgraph", False) or os.environ.get("USE_LANGGRAPH") == "1")
    basic_args = {
        "model_name_or_path": model_name_or_path,
    }
    total_cost = 0
    print(f"Filtered to {len(test_dataset)} instances")
    with open(output_file, "a+") as f:
        for datum in tqdm(test_dataset, desc=f"Inference for {model_name_or_path}"):
            instance_id = datum["instance_id"]
            if instance_id in existing_ids:
                continue
            output_dict = {"instance_id": instance_id}
            output_dict.update(basic_args)
            output_dict["text"] = f"{datum['text']}\n\n"
            if use_langgraph:
                print(f"Using 🔧LangGraph for {output_dict['model_name_or_path']}")
                response, cost = call_chat_via_langgraph(
                    output_dict["model_name_or_path"],
                    output_dict["text"],
                    use_azure,
                    temperature,
                    top_p,
                )
            else:
                response, cost = call_chat(
                    output_dict["model_name_or_path"],
                    output_dict["text"],
                    use_azure,
                    temperature,
                    top_p,
                )
            completion = response.choices[0].message.content
            total_cost += cost
            print(f"Total Cost: {total_cost:.2f}")
            output_dict["full_output"] = completion
            output_dict["model_patch"] = extract_diff(completion)
            print(json.dumps(output_dict), file=f, flush=True)
            if max_cost is not None and total_cost >= max_cost:
                print(f"Reached max cost {max_cost}, exiting")
                break


@retry(wait=wait_random_exponential(min=60, max=600), stop=stop_after_attempt(6))
def call_anthropic(
    inputs, anthropic, model_name_or_path, temperature, top_p, **model_args
):
    """
    Calls the anthropic API to generate completions for the given inputs.

    Args:
    inputs (str): The inputs to generate completions for.
    anthropic (Anthropic): The anthropic API object.
    model_name_or_path (str): The name or path of the model to use.
    temperature (float): The temperature to use.
    top_p (float): The top_p to use.
    model_args (dict): A dictionary of model arguments.
    """
    try:
        completion = anthropic.completions.create(
            model=model_name_or_path,
            max_tokens_to_sample=6000,
            prompt=inputs,
            temperature=temperature,
            top_p=top_p,
            **model_args,
        )
        response = completion.completion
        input_tokens = anthropic.count_tokens(inputs)
        output_tokens = anthropic.count_tokens(response)
        cost = calc_cost(model_name_or_path, input_tokens, output_tokens)
        return completion, cost
    except Exception as e:
        logger.error(e)
        logger.error(f"Inputs: {inputs}")
        traceback.print_exc()
        time.sleep(20)
        return None


@retry(wait=wait_random_exponential(min=60, max=600), stop=stop_after_attempt(6))
def call_anthropic_v2(
    inputs, anthropic, model_name_or_path, temperature, top_p, **model_args
):
    """
    Calls the anthropic API to generate completions for the given inputs.

    Args:
    inputs list(str): The inputs to generate completions for.
    anthropic (Anthropic): The anthropic API object.
    model_name_or_path (str): The name or path of the model to use.
    temperature (float): The temperature to use.
    top_p (float): The top_p to use.
    model_args (dict): A dictionary of model arguments.
    """
    system_messages = inputs.split("\n", 1)[0]
    user_message = inputs.split("\n", 1)[1]
    try:
        messages = [
            {"role": "user", "content": user_message},
        ]
        response = anthropic.messages.create(
            messages=messages,
            max_tokens=4096,
            model=model_name_or_path,
            temperature=temperature,
            top_p=top_p,
            system=system_messages,
        )
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calc_cost(response.model, input_tokens, output_tokens)
        return response, cost
    except Exception as e:
        logger.error(e)
        logger.error(f"Inputs: {inputs}")
        traceback.print_exc()
        time.sleep(20)
        return None


def anthropic_inference(
    test_dataset,
    model_name_or_path,
    output_file,
    model_args,
    existing_ids,
    max_cost,
):
    """
    Runs inference on a dataset using the anthropic API.

    Args:
    test_dataset (datasets.Dataset): The dataset to run inference on.
    model_name_or_path (str): The name or path of the model to use.
    output_file (str): The path to the output file.
    model_args (dict): A dictionary of model arguments.
    existing_ids (set): A set of ids that have already been processed.
    max_cost (float): The maximum cost to spend on inference.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", None)
    if api_key is None:
        raise ValueError(
            "Must provide an api key. Expected in ANTHROPIC_API_KEY environment variable."
        )
    print(f"Using Anthropic key {'*' * max(0, len(api_key) - 5) + api_key[-5:]}")
    anthropic = Anthropic(api_key=api_key)
    test_dataset = test_dataset.filter(
        lambda x: claude_tokenize(x["text"], anthropic)
        <= get_model_limit(model_name_or_path),
        desc="Filtering",
        load_from_cache_file=False,
    )
    temperature = model_args.pop("temperature", 0.2)
    top_p = model_args.pop("top_p", 0.95 if temperature > 0 else 1)
    print(f"Using temperature={temperature}, top_p={top_p}")
    basic_args = {
        "model_name_or_path": model_name_or_path,
    }
    total_cost = 0
    print(f"Filtered to {len(test_dataset)} instances")
    if "claude-3" in model_name_or_path.lower():
        call_api = call_anthropic_v2
    else:
        call_api = call_anthropic
    with open(output_file, "a+") as f:
        for datum in tqdm(test_dataset, desc=f"Inference for {model_name_or_path}"):
            instance_id = datum["instance_id"]
            if instance_id in existing_ids:
                continue
            output_dict = {"instance_id": instance_id}
            output_dict.update(basic_args)
            if "claude-3" in model_name_or_path.lower():
                output_dict["text_inputs"] = f"{datum['text']}\n"
            else:
                output_dict["text_inputs"] = (
                    f"{HUMAN_PROMPT} {datum['text']}\n\n{AI_PROMPT}"
                )
            try:
                completion, cost = call_api(
                    output_dict["text_inputs"],
                    anthropic,
                    model_name_or_path,
                    temperature,
                    top_p,
                    **model_args,
                )
            except Exception as e:
                logger.error(e)
                traceback.print_exc()
                continue
            total_cost += cost
            print(f"Total Cost: {total_cost:.2f}")
            if "claude-3" in model_name_or_path.lower():
                output_dict["full_output"] = completion.content[0].text
            else:
                output_dict["full_output"] = completion.completion
            output_dict["model_patch"] = extract_diff(output_dict["full_output"])
            print(json.dumps(output_dict), file=f, flush=True)
            if max_cost is not None and total_cost >= max_cost:
                print(f"Reached max cost {max_cost}, exiting")
                break


def parse_model_args(model_args):
    """
    Parses a string of model arguments and returns a dictionary of keyword arguments.

    Args:
        model_args (str): A string of comma-separated key-value pairs representing model arguments.

    Returns:
        dict: A dictionary of keyword arguments parsed from the input string.
    """
    kwargs = dict()
    if model_args is not None:
        for arg in model_args.split(","):
            key, value = arg.split("=")
            # infer value type
            if value in {"True", "False"}:
                kwargs[key] = value == "True"
            elif value.isnumeric():
                kwargs[key] = int(value)
            elif value.replace(".", "", 1).isnumeric():
                kwargs[key] = float(value)
            elif value in {"None"}:
                kwargs[key] = None
            elif value in {"[]"}:
                kwargs[key] = []
            elif value in {"{}"}:
                kwargs[key] = {}
            elif value.startswith("'") and value.endswith("'"):
                kwargs[key] = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                kwargs[key] = value[1:-1]
            else:
                kwargs[key] = value
    return kwargs


def main(
    dataset_name_or_path,
    split,
    model_name_or_path,
    shard_id,
    num_shards,
    output_dir,
    model_args,
    max_cost,
):
    if shard_id is None and num_shards is not None:
        logger.warning(
            f"Received num_shards={num_shards} but shard_id is None, ignoring"
        )
    if shard_id is not None and num_shards is None:
        logger.warning(f"Received shard_id={shard_id} but num_shards is None, ignoring")
    model_args = parse_model_args(model_args)
    model_nickname = model_name_or_path
    if "checkpoint" in Path(model_name_or_path).name:
        model_nickname = Path(model_name_or_path).parent.name
    else:
        model_nickname = Path(model_name_or_path).name
    output_file = f"{model_nickname}__{dataset_name_or_path.split('/')[-1]}__{split}"
    if shard_id is not None and num_shards is not None:
        output_file += f"__shard-{shard_id}__num_shards-{num_shards}"
    output_file = Path(output_dir, output_file + ".jsonl")
    logger.info(f"Will write to {output_file}")
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                instance_id = data["instance_id"]
                existing_ids.add(instance_id)
    logger.info(f"Read {len(existing_ids)} already completed ids from {output_file}")
    if Path(dataset_name_or_path).exists():
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)
    if split not in dataset:
        raise ValueError(f"Invalid split {split} for dataset {dataset_name_or_path}")
    dataset = dataset[split]
    lens = np.array(list(map(len, dataset["text"])))
    dataset = dataset.select(np.argsort(lens))
    if len(existing_ids) > 0:
        dataset = dataset.filter(
            lambda x: x["instance_id"] not in existing_ids,
            desc="Filtering out existing ids",
            load_from_cache_file=False,
        )
    if shard_id is not None and num_shards is not None:
        dataset = dataset.shard(num_shards, shard_id, contiguous=True)
    inference_args = {
        "test_dataset": dataset,
        "model_name_or_path": model_name_or_path,
        "output_file": output_file,
        "model_args": model_args,
        "existing_ids": existing_ids,
        "max_cost": max_cost,
    }
    # Prefer OpenRouter if key is provided regardless of model prefix (OpenAI-compatible endpoint)
    if os.environ.get("OPENROUTER_API_KEY"):
        openai_inference(**inference_args)
    else:
        if model_name_or_path.startswith("claude"):
            anthropic_inference(**inference_args)
        elif model_name_or_path.startswith("gpt"):
            openai_inference(**inference_args)
        else:
            raise ValueError(f"Invalid model name or path {model_name_or_path}")
    logger.info("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name of API model. Supports OpenRouter model ids when OPENROUTER_API_KEY is set.",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="Shard id to process. If None, process all shards.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Number of shards. If None, process all shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the output file.",
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default=None,
        help="List of model arguments separated by commas. (e.g. 'top_p=0.95,temperature=0.70')",
    )
    parser.add_argument(
        "--max_cost",
        type=float,
        default=None,
        help="Maximum cost to spend on inference.",
    )
    args = parser.parse_args()
    main(**vars(args))
