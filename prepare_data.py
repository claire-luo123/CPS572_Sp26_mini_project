import os
import json
from datasets import load_dataset

def get_content(row, keys):
    """Helper to find the right key in a dictionary."""
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None

def prepare_mixed_dataset():
    print("Loading Math Data (100%)...")
    math_ds = load_dataset("openai/gsm8k", "main", split="train")
    
    print("Loading & Subsampling IF Data...")
    if_ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    if_ds = if_ds.shuffle(seed=42).select(range(15000))
    
    print("Loading & Subsampling Code Data...")
    code_ds = load_dataset("nvidia/OpenCodeInstruct", split="train")
    code_ds = code_ds.shuffle(seed=42).select(range(15000))
    
    # Create the data directory safely
    os.makedirs("data", exist_ok=True)
    output_path = "data/my_mixed_training_data.jsonl"
    
    print(f"Saving all records to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Write Math records
        for row in math_ds:
            f.write(json.dumps({"question": row["question"], "answer": row["answer"]}) + "\n")
            
        # Write Instruction Following records
        for row in if_ds:
            f.write(json.dumps({"messages": row["messages"]}) + "\n")
            
        # Write Code records (handling variable column names)
        for row in code_ds:
            # Check for common input/output keys
            instr = get_content(row, ["instruction", "input", "prompt", "question"])
            out = get_content(row, ["output", "response", "answer", "content"])
            
            if instr and out:
                f.write(json.dumps({"instruction": instr, "output": out}) + "\n")
            
    print("Done! Dataset is prepped and ready for training.")

if __name__ == "__main__":
    prepare_mixed_dataset()