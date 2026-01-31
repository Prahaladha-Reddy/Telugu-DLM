"""
Prepare Telugu SFT data for Diffusion LM training.

Supports:
- maya-research/IndicVault (question/response format)
- Tensoic/GPTeacher-Telugu (instruction/input/output format)

This script mirrors the researcher's approach:
1. Load dataset
2. Train/test split
3. Tokenize with map()
4. Filter by length
5. Add query_mask with map()
6. Save to disk
"""

import argparse
import re
import time
from datasets import load_dataset, load_from_disk
from tokenizer import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Data Prep for Telugu Diffusion LM")
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="maya-research/IndicVault",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="Telugu",
        help="Dataset config/subset (e.g., 'Telugu' for IndicVault)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples (None for all)"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=256,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--test_split_pct",
        type=float,
        default=0.01,
        help="Percentage for test split"
    )
    parser.add_argument(
        "--path_to_data_store",
        type=str,
        required=True,
        help="Path to save processed data"
    )
    parser.add_argument(
        "--huggingface_cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--dataset_split_seed",
        type=int,
        default=42,
        help="Random seed for split"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for processing"
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        default="ai4bharat/IndicBERTv2-MLM-only",
        help="Model name for tokenizer"
    )
    # For backward compatibility with old arg names
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    
    return parser.parse_args()


def is_valid_response(response: str) -> bool:
    """Filter out error responses and empty strings."""
    if not response or not response.strip():
        return False
    
    # Filter API errors that appear in IndicVault
    error_patterns = [
        r"ERROR:",
        r"No API keys available",
        r"rate limit",
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return False
    
    return True


def prepare_data(args):
    """
    Process dataset for SFT training.
    
    Following the researcher's exact pattern from his data_prep.py
    """
    
    # Handle backward compatible args
    context_length = args.max_length if args.max_length else args.context_length
    seed = args.seed if args.seed else args.dataset_split_seed
    path_to_save = args.path_to_data_store
    cache_dir = args.huggingface_cache_dir
    
    print("=" * 60)
    print("PREPARING TELUGU SFT DATA")
    print("=" * 60)
    
    # =========================================================================
    # Load tokenizer
    # =========================================================================
    print(f"\n[1/7] Loading tokenizer: {args.hf_model_name}")
    tokenizer = get_tokenizer(args.hf_model_name)
    print(f"      Vocab size: {len(tokenizer)}")
    print(f"      PAD == EOS: {tokenizer.pad_token_id == tokenizer.eos_token_id}")
    
    # =========================================================================
    # Load dataset
    # =========================================================================
    print(f"\n[2/7] Loading dataset: {args.dataset_name}")
    
    try:
        if args.dataset_config:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config,
                split="train",
                cache_dir=cache_dir,
            )
        else:
            dataset = load_dataset(
                args.dataset_name,
                split="train",
                cache_dir=cache_dir,
            )
    except Exception as e:
        print(f"      Error with config, trying without: {e}")
        dataset = load_dataset(
            args.dataset_name,
            split="train",
            cache_dir=cache_dir,
        )
    
    print(f"      Loaded: {len(dataset)} samples")
    print(f"      Columns: {dataset.column_names}")
    
    # Detect format
    sample = dataset[0]
    if "question" in sample and "response" in sample:
        format_type = "indicvault"
    elif "instruction" in sample:
        format_type = "alpaca"
    else:
        raise ValueError(f"Unknown dataset format. Columns: {dataset.column_names}")
    
    print(f"      Format: {format_type}")
    
    # =========================================================================
    # Filter invalid responses (for IndicVault which has API errors)
    # =========================================================================
    if format_type == "indicvault":
        print(f"\n[3/7] Filtering invalid responses...")
        original_len = len(dataset)
        dataset = dataset.filter(
            lambda x: is_valid_response(x.get("response", "")),
            num_proc=args.num_workers,
        )
        print(f"      Kept: {len(dataset)} / {original_len} samples")
    else:
        print(f"\n[3/7] Skipping filter (not IndicVault)")
    
    # =========================================================================
    # Limit samples if requested
    # =========================================================================
    if args.max_samples and len(dataset) > args.max_samples:
        print(f"\n[3.5/7] Limiting to {args.max_samples} samples...")
        dataset = dataset.shuffle(seed=seed).select(range(args.max_samples))
        print(f"      Selected: {len(dataset)} samples")
    
    # =========================================================================
    # Train/Test Split - BEFORE tokenization (like researcher does)
    # =========================================================================
    print(f"\n[4/7] Train/test split ({args.test_split_pct*100:.1f}% test)")
    dataset = dataset.train_test_split(
        test_size=args.test_split_pct, 
        seed=seed
    )
    print(f"      Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    
    # =========================================================================
    # Define chat template function (like researcher's apply_chat_template)
    # =========================================================================
    def apply_chat_template(query, response):
        """Apply chat template and return list of token IDs."""
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ],
            tokenize=True,
            add_special_tokens=True,
        )
    
    # =========================================================================
    # Preprocess function (like researcher's preprocess)
    # =========================================================================
    def preprocess(example):
        """Tokenize a single example."""
        if format_type == "indicvault":
            question = example["question"]
            answer = example["response"]
        else:  # alpaca
            instruction = example["instruction"]
            inp = example.get("input", "")
            answer = example["output"]
            
            # Combine like researcher does
            if inp and len(inp.strip()) > 0:
                question = instruction.rstrip(".") + ": " + inp
            else:
                question = instruction
        
        # Apply chat template - returns list of ints
        result  = apply_chat_template(question, answer)
        tokenized = result["input_ids"]

        return {"input_ids": tokenized, "length": len(tokenized)}
    
    # =========================================================================
    # Tokenize with map() - exactly like researcher
    # =========================================================================
    print(f"\n[5/7] Tokenizing...")
    
    # Get columns to remove
    if format_type == "indicvault":
        cols_to_remove = ["question", "response"]
    else:
        cols_to_remove = ["instruction", "input", "output"]
    
    # Only remove columns that exist
    cols_to_remove = [c for c in cols_to_remove if c in dataset["train"].column_names]
    
    tokenized_data = dataset.map(
        preprocess,
        num_proc=args.num_workers,
        remove_columns=cols_to_remove,
    )
    
    # =========================================================================
    # Filter by length - exactly like researcher
    # =========================================================================
    print(f"\n[6/7] Filtering by length (max={context_length})...")
    
    def keep_within_context(example):
        return example["length"] <= context_length
    
    before_train = len(tokenized_data["train"])
    before_test = len(tokenized_data["test"])
    
    tokenized_data = tokenized_data.filter(
        keep_within_context, 
        num_proc=args.num_workers
    )
    tokenized_data = tokenized_data.remove_columns("length")
    
    print(f"      Train: {before_train} -> {len(tokenized_data['train'])}")
    print(f"      Test: {before_test} -> {len(tokenized_data['test'])}")
    
    # =========================================================================
    # Add query_mask - exactly like researcher's get_answer_mask
    # =========================================================================
    print(f"\n[7/7] Adding query_mask...")
    
    def get_answer_mask(example):
        """
        Create query_mask where 1 = answer region.
        Exactly matching researcher's implementation.
        """
        tokenized = example["input_ids"]
        
        query_mask = []
        occurrence = 0
        is_answer = False
        
        for t in tokenized:
            check = (t == tokenizer.convert_tokens_to_ids("<END_ID>"))
            
            if not is_answer:
                query_mask.append(0)
            else:
                query_mask.append(1)
            
            if check:
                if occurrence == 0:
                    occurrence += 1
                else:
                    is_answer = True
        
        example["query_mask"] = query_mask
        return example
    
    tokenized_data = tokenized_data.map(
        get_answer_mask,
        num_proc=args.num_workers,
    )
    
    # =========================================================================
    # Save - exactly like researcher
    # =========================================================================
    print(f"\n[SAVING] to: {path_to_save}")
    tokenized_data.save_to_disk(path_to_save)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Train samples: {len(tokenized_data['train'])}")
    print(f"Test samples: {len(tokenized_data['test'])}")
    
    # Show a sample
    sample = tokenized_data["train"][0]
    input_ids = sample["input_ids"]
    query_mask = sample["query_mask"]
    
    print(f"\nSample:")
    print(f"  Sequence length: {len(input_ids)}")
    print(f"  Answer region length: {sum(query_mask)}")
    print(f"  First 10 tokens: {input_ids[:10]}")
    print(f"  First 10 mask: {query_mask[:10]}")
    
    print(f"\nDecoded:")
    print(tokenizer.decode(input_ids, skip_special_tokens=False)[:500])
    
    print(f"\nAnswer region:")
    answer_tokens = [t for t, m in zip(input_ids, query_mask) if m == 1]
    if answer_tokens:
        print(tokenizer.decode(answer_tokens, skip_special_tokens=False)[:300])
    else:
        print("  (empty - check query_mask logic)")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    prepare_data(args)
    
    # Verify it loads correctly (like researcher does)
    print("\n[VERIFICATION] Loading saved data...")
    start = time.time()
    data = load_from_disk(args.path_to_data_store)
    end = time.time()
    print(f"Time to load: {end - start:.2f}s")
    print(f"Dataset: {data}")
    
    # Test that collator will work
    print("\n[COLLATOR TEST]")
    sample = data["train"][0]
    print(f"  input_ids type: {type(sample['input_ids'])}")
    print(f"  query_mask type: {type(sample['query_mask'])}")
    
    import torch
    try:
        t = torch.tensor(sample["input_ids"])
        print(f"  ✓ Can convert to tensor: shape={t.shape}")
    except Exception as e:
        print(f"  ✗ Cannot convert to tensor: {e}")