"""
Data utilities for Telugu Diffusion LM.

Key insight: 
- pad_sequence with padding_value=1 for query_mask
- This means PAD positions are included in the answer region
- Model learns to predict PAD/EOS after the answer
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizer import get_tokenizer


def SFTCollator(model_name="ai4bharat/IndicBERTv2-MLM-only"):
    """
    Collator for SFT training.
    
    Key: query_mask is padded with 1s (not 0s!)
    This means PAD positions are part of the "answer region" and can be masked.
    Since PAD == EOS, model learns to predict EOS after the answer.
    """
    tokenizer = get_tokenizer(model_name)
    eos_token_id = tokenizer.eos_token_id
    
    def _collate_fn(batch):
        # Convert to tensors
        inputs = [torch.tensor(b["input_ids"]) for b in batch]
        query_masks = [torch.tensor(b["query_mask"]) for b in batch]
        
        # Pad sequences
        # input_ids: pad with EOS (which is same as PAD)
        # query_mask: pad with 1 (so PADs are included in answer region!)
        inputs = pad_sequence(inputs, padding_value=eos_token_id, batch_first=True)
        query_masks = pad_sequence(query_masks, padding_value=1, batch_first=True)
        
        return {
            "input_ids": inputs,
            "query_mask": query_masks
        }
    
    return _collate_fn


if __name__ == "__main__":
    # Test the collator
    from datasets import load_from_disk
    from torch.utils.data import DataLoader
    
    # You would replace this with your actual data path
    # data = load_from_disk("./data/telugu_sft")["train"]
    
    # Create some dummy data for testing
    tok = get_tokenizer()
    
    samples = [
        {
            "input_ids": [1, 2, 3, 4, 5, tok.eos_token_id],
            "query_mask": [0, 0, 1, 1, 1, 1]
        },
        {
            "input_ids": [1, 2, 3, tok.eos_token_id],
            "query_mask": [0, 1, 1, 1]
        },
    ]
    
    collator = SFTCollator()
    batch = collator(samples)
    
    print("input_ids shape:", batch["input_ids"].shape)
    print("query_mask shape:", batch["query_mask"].shape)
    print("\ninput_ids:")
    print(batch["input_ids"])
    print("\nquery_mask (note: PADs have mask=1!):")
    print(batch["query_mask"])
