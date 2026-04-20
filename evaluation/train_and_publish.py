"""
Train a model (minimal SFT), save checkpoint, and publish it.

NOTE: This is a TOY EXAMPLE that trains for a few steps on dummy data
to verify the full workflow end-to-end. You should replace the training
data and training logic with your own implementation.

TODO:
  - Replace DEMO_CONVERSATIONS with your task-specific training data
  - Tune hyperparameters (learning rate, batch size, number of steps, LoRA rank)
  - Add validation / early stopping as needed

Usage:
    python evaluation/train_and_publish.py
    python evaluation/train_and_publish.py --num_steps 20
    python evaluation/train_and_publish.py --no_publish   # skip publishing
"""

import argparse
import json
import os

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Default to 1B for cheaper iteration; switch to 3B/8B when recipe stabilizes.
MODEL = "meta-llama/Llama-3.2-1B"
# MODEL = "meta-llama/Llama-3.2-3B"
# MODEL = "meta-llama/Llama-3.1-8B"    # Recommended for final submission

# Longer contexts help code / long IF rows; raise carefully if you hit memory limits.
MAX_SEQ_LEN = 1024

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Train, save, and publish a checkpoint")
    parser.add_argument("--num_steps", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--checkpoint_name", type=str, default="mixed_sft_1b", help="Checkpoint name")
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    args = parser.parse_args()

    # Setup
    print(f"Model: {MODEL}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Load custom training data
    print("Loading custom training data from JSONL...")
    all_conversations = []
    
    # Update this path if your jsonl file is named differently!
    data_path = os.path.join(EVAL_DIR, "..", "data", "my_mixed_training_data.jsonl") 
    
    with open(data_path, "r") as f:
        for line in f:
            row = json.loads(line)
            # Map GSM8K format
            if "question" in row and "answer" in row:
                all_conversations.append([
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]}
                ])
            # Map Tulu format
            elif "messages" in row:
                all_conversations.append(row["messages"])
            # Map Code format (Instruction/Output)
            elif "instruction" in row and "output" in row:
                all_conversations.append([
                    {"role": "user", "content": row["instruction"]},
                    {"role": "assistant", "content": row["output"]}
                ])

    # Prepare training data
    print(f"Tokenizing training data (max_length={MAX_SEQ_LEN})...")
    all_data = []
    for convo in all_conversations:
        try:
            datum = conversation_to_datum(
                convo,
                renderer,
                max_length=MAX_SEQ_LEN,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            all_data.append(datum)
        except Exception as e:
            continue # Skip any malformed data rows
            
    print(f"  {len(all_data)} training examples prepared and tokenized!")

    # Create training client
    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=MODEL, rank=args.rank)
    print("  Training client ready")

    # Train
    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr})...")

    for step in range(args.num_steps):
        # Cycle through data
        start = (step * args.batch_size) % len(all_data)
        batch = [all_data[i % len(all_data)] for i in range(start, start + args.batch_size)]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        # Compute loss
        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        
        # Only print every 10 steps so it doesn't spam your terminal
        if step % 10 == 0 or step == args.num_steps - 1:
            print(f"  Step {step+1}/{args.num_steps} | Loss: {loss:.4f}")

    # Save checkpoint
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

    # Publish
    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published successfully!")
    else:
        print("\nSkipping publish (--no_publish).")

    # Save checkpoint info
    info = {
        "checkpoint_path": checkpoint_path,
        "base_model": MODEL,
        "renderer_name": renderer_name,
        "training": {
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lora_rank": args.rank,
            "max_seq_len": MAX_SEQ_LEN,
        },
        "published": not args.no_publish,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nNext: evaluate your checkpoint with")
    print(f"  PYTHONPATH=. python evaluation/eval_all.py --checkpoint_path \"{checkpoint_path}\" --base_model {MODEL}")

if __name__ == "__main__":
    main()
