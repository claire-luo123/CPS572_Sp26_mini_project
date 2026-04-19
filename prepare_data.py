from datasets import load_dataset, concatenate_datasets

def prepare_mixed_dataset():
    print("Loading Math Data (100%)...")
    # We take the entire train split (7,473 samples)
    math_ds = load_dataset("openai/gsm8k", "main", split="train")
    
    print("Loading & Subsampling IF Data...")
    # Tulu is huge (939k). We shuffle it and take 15,000 samples.
    if_ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    if_ds = if_ds.shuffle(seed=42).select(range(15000))
    
    print("Loading & Subsampling Code Data...")
    # OpenCodeInstruct is 5M+. We shuffle and take 15,000 samples.
    # EXTENSION IDEA: Write a filter function here to only grab Python examples!
    code_ds = load_dataset("nvidia/OpenCodeInstruct", split="train")
    code_ds = code_ds.shuffle(seed=42).select(range(15000))
    
    # NOTE: Before concatenating, you will need to map all three datasets 
    # to have the exact same column names/formats expected by Tinker 
    # (usually a "messages" or "text" column). 
    
    print("Saving mixed dataset locally...")
    # Save the processed data to a local file so your training script can load it instantly
    # mixed_ds.to_json("data/my_mixed_training_data.jsonl")
    print("Done!")

if __name__ == "__main__":
    prepare_mixed_dataset()