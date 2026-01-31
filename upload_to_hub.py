"""
Reconstruct the Telugu Diffusion LM from SafeTensors and upload to HuggingFace.

Steps:
1. Download the safetensors file from Google Drive
2. Load base model architecture with resized embeddings
3. Load the trained weights
4. Save with proper tokenizer and config
5. Upload to HuggingFace Hub
"""

import os
import torch
from transformers import AutoModelForMaskedLM, AutoConfig
from safetensors.torch import load_file
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer


# ============================================================
# STEP 1: Define the exact tokenizer used during training
# ============================================================

def get_tokenizer(
    model_name: str = "ai4bharat/IndicBERTv2-MLM-only",
    bos_token: str = "<BOS>",
    eos_token: str = "<EOS>",
    start_token: str = "<START_ID>",
    end_token: str = "<END_ID>",
    eot_token: str = "<EOT_ID>",
):
    """Exact tokenizer from your training code."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    special_tokens = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "additional_special_tokens": [start_token, end_token, eot_token],
    }
    tokenizer.add_special_tokens(special_tokens)

    tokenizer.pad_token = eos_token
    tokenizer.cls_token = bos_token

    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        special_tokens=[
            (bos_token, tokenizer.bos_token_id),
            (eos_token, tokenizer.eos_token_id),
        ],
    )

    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ bos_token if loop.first else '' }}"
        f"{{{{ '{start_token}' + message['role'] + '{end_token}' }}}}\n"
        "{{ message['content'] }}"
        f"{{{{ '{eot_token}' if message['role'] == 'user' else eos_token }}}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        f"{{{{ '{start_token}' + 'assistant' + '{end_token}' }}}}"
        "{% endif %}"
    )

    return tokenizer


# ============================================================
# STEP 2: Reconstruct the model
# ============================================================

def reconstruct_model(
    safetensors_path: str,
    output_dir: str,
    base_model_name: str = "ai4bharat/IndicBERTv2-MLM-only",
):
    """
    Reconstruct complete HF model from safetensors weights.
    
    Args:
        safetensors_path: Path to the downloaded .safetensors file
        output_dir: Where to save the complete model
        base_model_name: The base model used for architecture
    """
    print("=" * 60)
    print("RECONSTRUCTING TELUGU DIFFUSION LM")
    print("=" * 60)
    
    # 1. Get tokenizer with custom tokens
    print("\n[1/5] Loading tokenizer with custom special tokens...")
    tokenizer = get_tokenizer(base_model_name)
    print(f"      Tokenizer vocab size: {len(tokenizer)}")
    
    # 2. Load base model and resize embeddings
    print("\n[2/5] Loading base model architecture...")
    model = AutoModelForMaskedLM.from_pretrained(base_model_name)
    
    # Resize embeddings to match tokenizer (this adds the 5 special tokens)
    original_vocab = model.config.vocab_size
    model.resize_token_embeddings(len(tokenizer))
    print(f"      Resized embeddings: {original_vocab} -> {len(tokenizer)}")
    
    # 3. Load trained weights
    print("\n[3/5] Loading trained weights from safetensors...")
    state_dict = load_file(safetensors_path)
    
    # Load with strict=False to handle any minor mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"      Warning - Missing keys: {missing[:5]}..." if len(missing) > 5 else f"      Missing: {missing}")
    if unexpected:
        print(f"      Warning - Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"      Unexpected: {unexpected}")
    
    # Tie weights (important for MLM models)
    model.tie_weights()
    print("      Weights loaded and tied successfully!")
    
    # 4. Update config with important metadata
    print("\n[4/5] Updating model config...")
    model.config.vocab_size = len(tokenizer)
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Add custom metadata to config
    model.config.custom_model_type = "telugu-diffusion-lm"
    model.config.base_model = base_model_name
    
    # 5. Save everything
    print(f"\n[5/5] Saving complete model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_dir, safe_serialization=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "=" * 60)
    print("RECONSTRUCTION COMPLETE!")
    print("=" * 60)
    print(f"\nSaved files in {output_dir}:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f)) / (1024*1024)
        print(f"  - {f} ({size:.1f} MB)")
    
    return model, tokenizer


# ============================================================
# STEP 3: Upload to HuggingFace Hub
# ============================================================

def upload_to_hub(
    output_dir: str,
    repo_id: str,
    private: bool = False,
):
    """
    Upload the reconstructed model to HuggingFace Hub.
    
    Args:
        output_dir: Local directory with saved model
        repo_id: HuggingFace repo (e.g., "your-username/telugu-diffusion-lm")
        private: Whether to make the repo private
    """
    from huggingface_hub import HfApi, create_repo
    
    print(f"\nUploading to HuggingFace Hub: {repo_id}")
    
    # Create repo if it doesn't exist
    api = HfApi()
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"  Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Upload all files
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        commit_message="Upload Telugu Diffusion LM (SFT trained)"
    )
    
    print(f"\n✅ Upload complete!")
    print(f"   View your model: https://huggingface.co/{repo_id}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # ----- CONFIGURATION -----
    # Update these paths!
    
    SAFETENSORS_PATH = "./model.safetensors"  # Path to downloaded file
    OUTPUT_DIR = "./telugu-diffusion-lm-reconstructed"
    HF_REPO_ID = "Prahaladha/telugu-diffusion-lm"  # Change this!
    
    # ----- RUN -----
    
    # Step 1: Reconstruct
    model, tokenizer = reconstruct_model(
        safetensors_path=SAFETENSORS_PATH,
        output_dir=OUTPUT_DIR,
    )
    
    # Step 2: Quick verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Test tokenization
    test_text = "నమస్కారం, నేను తెలుగు డిఫ్యూజన్ మోడల్ ని"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"\nTest tokenization:")
    print(f"  Input: {test_text}")
    print(f"  Token IDs: {tokens['input_ids'][0][:10].tolist()}...")
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(**tokens)
    print(f"  Forward pass: ✅ (logits shape: {outputs.logits.shape})")
    
    # Step 3: Upload (uncomment when ready)
    # upload_to_hub(OUTPUT_DIR, HF_REPO_ID, private=False)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Download the safetensors from Google Drive
2. Update SAFETENSORS_PATH above
3. Run this script to reconstruct
4. Uncomment upload_to_hub() and set your HF_REPO_ID
5. Run again to upload

To download from Google Drive via command line:
    pip install gdown
    gdown "16KxNLf4OY8PPfyISkW8SO1BqLO7zClDH"
""")