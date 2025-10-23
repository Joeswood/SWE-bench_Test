#!/usr/bin/env python3
# copy the prompt functions from create_instance.py
from argparse import ArgumentParser
from swebench.inference.make_datasets.create_instance import (
    PROMPT_FUNCTIONS, make_code_text, PATCH_EXAMPLE
)
from swebench.inference.make_datasets.create_text_dataset import main as create_text_main

CONSISTENT_INSTRUCTION = "Ensure that your response strictly matches the format above."

def prompt_style_3_cot(instance, consistent=True):
    premise = "You will be provided with a partial code base and an issue statement explaining a problem to resolve."
    cot_prompt = "Please think step by step and provide your solution."
    readmes_text = make_code_text(instance["readmes"], add_line_numbers=False)
    code_text = make_code_text(instance["file_contents"], add_line_numbers=False)
    example_explanation = (
        "Here is an example of a patch file. It consists of changes to the code base. "
        + "It specifies the file names, the line numbers of each change, and the removed and added lines. "
        + "A single patch file can contain changes to multiple files."
    )
    final_instruction = (
        "I need you to solve the provided issue by generating a single patch file that I can apply "
        + "directly to this repository using git apply. Please respond with a single patch "
        + "file in the format shown above."
    )
    problem_statement = instance["problem_statement"]
    final_text = [
        premise,
        "<issue>",
        problem_statement,
        "</issue>",
        "",
        "<code>",
        readmes_text,
        code_text,
        "</code>",
        "",
        example_explanation,
        "<patch>",
        PATCH_EXAMPLE,
        "</patch>",
        "",
        final_instruction,
        cot_prompt, # add CoT prompt
        "Respond below:",
        CONSISTENT_INSTRUCTION if consistent else "",
    ]
    final_text = "\n".join(final_text)
    return final_text


def prompt_style_3_self_repair(instance, consistent=True):
    premise = "You will be provided with a partial code base and an issue statement explaining a problem to resolve."
    self_repair_prompt = "Please self-repair the code base and provide the patch file." # self-repair prompt
    readmes_text = make_code_text(instance["readmes"], add_line_numbers=False)
    code_text = make_code_text(instance["file_contents"], add_line_numbers=False)
    example_explanation = (
        "Here is an example of a patch file. It consists of changes to the code base. "
        + "It specifies the file names, the line numbers of each change, and the removed and added lines. "
        + "A single patch file can contain changes to multiple files."
    )
    final_instruction = (
        "I need you to solve the provided issue by generating a single patch file that I can apply "
        + "directly to this repository using git apply. Please respond with a single patch "
        + "file in the format shown above."
    )
    problem_statement = instance["problem_statement"]
    final_text = [
        premise,
        "<issue>",
        problem_statement,
        "</issue>",
        "",
        "<code>",
        readmes_text,
        code_text,
        "</code>",
        "",
        example_explanation,
        "<patch>",
        PATCH_EXAMPLE,
        "</patch>",
        "",
        final_instruction,
        self_repair_prompt, # add self-repair prompt
        CONSISTENT_INSTRUCTION if consistent else "",
        "Respond below:",
    ]
    final_text = "\n".join(final_text)
    return final_text

# register the new prompt style
PROMPT_FUNCTIONS["style-3-cot"] = prompt_style_3_cot
PROMPT_FUNCTIONS["style-3-self-repair"] = prompt_style_3_self_repair

if __name__ == "__main__":
    # create the text dataset with cot prompt
    parser = ArgumentParser()
    parser.add_argument("--subset_type", type=str, required=True, choices=["10", "2"])
    parser.add_argument("--consistent", type=str, required=True, choices=["true", "false"])

    args = parser.parse_args()
    if args.subset_type == "10":
        subset_name = "subset_10"
    elif args.subset_type == "2":
        subset_name = "subset_2"
    else:
        raise ValueError(f"Invalid subset type: {args.subset_type}")

    # parse the consistent argument
    if args.consistent == "true":
        consistent = True
    elif args.consistent == "false":
        consistent = False

    create_text_main(
        dataset_name_or_path=f"./subset_10_swebench/{subset_name}",
        splits=["test"],
        validation_ratio=0.0,
        output_dir="./subset_10_swebench/text_ds/cot",
        retrieval_file=f"./subset_10_swebench/retrieval/{subset_name}/file_name_and_contents.retrieval.jsonl",
        prompt_style="style-3-cot",   # Use custom prompt style cot
        file_source="bm25",
        k=8,
        max_context_len=100000,
        tokenizer_name="cl100k",
        push_to_hub_user=None,
    )
    # create the text dataset with self-repair prompt
    create_text_main(
        dataset_name_or_path=f"./subset_10_swebench/{subset_name}",
        splits=["test"],
        validation_ratio=0.0,
        output_dir="./subset_10_swebench/text_ds/self_repair",
        retrieval_file=f"./subset_10_swebench/retrieval/{subset_name}/file_name_and_contents.retrieval.jsonl",
        prompt_style="style-3-self-repair",   # Use custom prompt style self-repair
        file_source="bm25",
        k=8,
        max_context_len=100000,
        tokenizer_name="cl100k",
        push_to_hub_user=None,
    )
    