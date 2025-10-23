# save_subset.py
from datasets import load_dataset, DatasetDict
from argparse import ArgumentParser

# choose first 10 instance_id from SWE-bench_Verified
# https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/viewer?views%5B%5D=test&sql=SELECT+instance_id+%0AFROM+test+%0ALIMIT+10%3B
ids = {"astropy__astropy-12907",
        "astropy__astropy-13033",
        "astropy__astropy-13236",       
        "astropy__astropy-13398",
        "astropy__astropy-13453",
        "astropy__astropy-13579",
        "astropy__astropy-13977",
        "astropy__astropy-14096",
        "astropy__astropy-14182",       
        "astropy__astropy-14309",
} 

refine_ids = {"astropy__astropy-13236", "astropy__astropy-14182"}

def save_subset():
    ds = load_dataset("SWE-bench/SWE-bench_Verified")           # or "SWE-bench/SWE-bench"
    subset = DatasetDict({split: split_ds.filter(lambda x: x["instance_id"] in ids)
                        for split, split_ds in ds.items()})
    subset.save_to_disk("./subset_10_swebench/subset_10")  # local dataset directory

def save_subset_2():
    ds = load_dataset("SWE-bench/SWE-bench_Verified")
    subset_2 = DatasetDict({split: split_ds.filter(lambda x: x["instance_id"] in refine_ids)
                        for split, split_ds in ds.items()})
    subset_2.save_to_disk("./subset_10_swebench/subset_2")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--subset_type", type=str, required=True, choices=["10", "2"])
    args = parser.parse_args()
    if args.subset_type == "10":
        save_subset()
    elif args.subset_type == "2":
        save_subset_2()
    else:
        raise ValueError(f"Invalid subset type: {args.subset_type}")
