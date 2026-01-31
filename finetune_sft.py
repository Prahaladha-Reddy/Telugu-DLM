"""
Telugu Diffusion LM - SFT Training Script

Matches the researcher's approach:
1. Dynamic batching with SFTCollator (pads to max in batch)
2. query_mask extended to include PAD positions by collator
3. Model learns to predict PAD/EOS after the answer
4. Loss scaled by 1/t for importance sampling

Key insight: Since PAD == EOS, the model just needs to learn
"after the answer content, keep predicting EOS/PAD"
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, get_scheduler
from datasets import load_from_disk
from accelerate import Accelerator
from tqdm import tqdm
from safetensors.torch import load_file, save_file

from tokenizer import get_tokenizer
from data_utils import SFTCollator


def parse_args():
    p = argparse.ArgumentParser("Telugu Diffusion LM - SFT Training")
    
    # Experiment
    p.add_argument("--experiment_name", required=True, type=str)
    p.add_argument("--working_directory", required=True, type=str)
    
    # Model
    p.add_argument("--hf_model_name", default="ai4bharat/IndicBERTv2-MLM-only", type=str)
    p.add_argument("--path_to_pretrained_checkpoint", default=None, type=str,
                   help="Optional: path to pretrained diffusion checkpoint")
    
    # Data
    p.add_argument("--path_to_prepped_data", required=True, type=str)
    p.add_argument("--num_workers", default=4, type=int)
    
    # Training
    p.add_argument("--per_gpu_batch_size", default=16, type=int)
    p.add_argument("--gradient_accumulation_steps", default=1, type=int)
    p.add_argument("--num_training_steps", default=15000, type=int)
    p.add_argument("--learning_rate", default=1e-5, type=float)
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--max_grad_norm", default=1.0, type=float)
    p.add_argument("--lr_scheduler_type", default="cosine", type=str)
    p.add_argument("--num_warmup_steps", default=500, type=int)
    
    # Logging
    p.add_argument("--logging_steps", default=25, type=int)
    p.add_argument("--evaluation_interval", default=1500, type=int)
    p.add_argument("--checkpoint_interval", default=3000, type=int)
    p.add_argument("--log_wandb", default=False, action=argparse.BooleanOptionalAction)
    
    return p.parse_args()


def main():
    args = parse_args()
    import os
    # Setup
    exp_dir = os.path.join(args.working_directory, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Accelerator
    accelerator = Accelerator(
        project_dir=exp_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.log_wandb else None,
    )
    
    if args.log_wandb:
        accelerator.init_trackers(args.experiment_name)
    
    # ========================
    # SETUP
    # ========================
    accelerator.print("=" * 60)
    accelerator.print("TELUGU DIFFUSION LM - SFT TRAINING")
    accelerator.print("=" * 60)
    
    # Tokenizer
    accelerator.print(f"\n[1/5] Loading tokenizer")
    tokenizer = get_tokenizer(args.hf_model_name)
    accelerator.print(f"      Vocab size: {len(tokenizer)}")
    accelerator.print(f"      EOS == PAD: {tokenizer.eos_token_id == tokenizer.pad_token_id}")
    
    # Model
    accelerator.print(f"\n[2/5] Loading model")
    
    if args.path_to_pretrained_checkpoint:
        # Load from local pretrained checkpoint (e.g., your MLM pretrained model)
        import os
        ckpt_path = args.path_to_pretrained_checkpoint
        
        if os.path.isdir(ckpt_path):
            # It's a directory - look for model.safetensors inside
            safetensors_file = os.path.join(ckpt_path, "model.safetensors")
        else:
            # It's a direct path to safetensors file
            safetensors_file = ckpt_path
        
        accelerator.print(f"      Loading from pretrained: {safetensors_file}")
        
        # Load base architecture first
        model = AutoModelForMaskedLM.from_pretrained(args.hf_model_name)
        model.resize_token_embeddings(len(tokenizer))
        
        # Load pretrained weights
        state_dict = load_file(safetensors_file)
        
        # Report what we're loading
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            accelerator.print(f"      Missing keys: {missing[:3]}..." if len(missing) > 3 else f"      Missing: {missing}")
        if unexpected:
            accelerator.print(f"      Unexpected keys: {unexpected[:3]}..." if len(unexpected) > 3 else f"      Unexpected: {unexpected}")
        
        model.tie_weights()
        accelerator.print(f"      ✓ Loaded pretrained weights successfully")
    else:
        # Load fresh from HuggingFace
        accelerator.print(f"      Loading fresh from: {args.hf_model_name}")
        model = AutoModelForMaskedLM.from_pretrained(args.hf_model_name)
        model.resize_token_embeddings(len(tokenizer))
    
    params = sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)
    accelerator.print(f"      Parameters: {params:,}")
    
    # Data
    accelerator.print(f"\n[3/5] Loading data: {args.path_to_prepped_data}")
    data = load_from_disk(args.path_to_prepped_data)
    accelerator.print(f"      Train: {len(data['train'])}, Test: {len(data['test'])}")
    
    # Collator - THIS IS KEY!
    # The collator pads sequences AND extends query_mask to include PAD positions
    collator = SFTCollator(args.hf_model_name)
    
    mini_bs = args.per_gpu_batch_size // args.gradient_accumulation_steps
    train_dl = DataLoader(
        data["train"], 
        batch_size=mini_bs, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    eval_dl = DataLoader(
        data["test"], 
        batch_size=mini_bs, 
        shuffle=False, 
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    
    # Optimizer
    accelerator.print(f"\n[4/5] Setting up optimizer")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
    )
    
    # Loss
    loss_func = nn.CrossEntropyLoss(reduction="none")
    
    # Prepare
    model, optimizer, train_dl, eval_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, eval_dl, scheduler
    )
    
    # Summary
    accelerator.print(f"\n[5/5] Training config:")
    accelerator.print(f"      Steps: {args.num_training_steps}")
    accelerator.print(f"      Batch: {args.per_gpu_batch_size} x {accelerator.num_processes} GPUs")
    accelerator.print(f"      Grad accum: {args.gradient_accumulation_steps}")
    accelerator.print(f"      LR: {args.learning_rate}")
    
    accelerator.print("\n" + "=" * 60)
    accelerator.print("STARTING TRAINING")
    accelerator.print("=" * 60 + "\n")
    
    # ========================
    # TRAINING LOOP
    # ========================
    model.train()
    completed = 0
    accum_loss = 0.0
    accum_steps = 0
    best_val_loss = float('inf')
    
    progress = tqdm(range(args.num_training_steps), disable=not accelerator.is_local_main_process)
    
    while completed < args.num_training_steps:
        for batch in train_dl:
            input_ids = batch["input_ids"].to(accelerator.device)
            query_mask = batch["query_mask"].to(accelerator.device).float()  # Convert to float for masking
            
            bsz, seq_len = input_ids.shape
            
            # Attend to all tokens
            attention_mask = torch.ones((bsz, seq_len), dtype=torch.long, device=accelerator.device)
            
            # ========================
            # DIFFUSION MASKING
            # ========================
            # Sample t ~ Uniform(0, 1) per sample, expand to all positions
            t = torch.rand(bsz, 1, device=accelerator.device)
            t = t.expand(bsz, seq_len).clamp_min(1e-5)
            
            # Bernoulli mask with probability t
            mask = torch.bernoulli(t)
            
            # Only mask where query_mask == 1 (answer region + PAD positions)
            # This is the KEY: since collator sets query_mask=1 for PADs,
            # the model will learn to predict PAD when those positions are masked
            mask = mask * query_mask
            mask = mask.bool()
            
            # Apply masking
            masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id)
            labels = input_ids.masked_fill(~mask, -100)
            
            # Forward
            logits = model(input_ids=masked_input_ids, attention_mask=attention_mask).logits
            
            # Loss (per token)
            loss = loss_func(
                logits.view(bsz * seq_len, -1),
                labels.view(-1)
            )
            loss = loss.view(bsz, seq_len)
            
            # Scale by 1/t (importance sampling for diffusion)
            loss = loss / t
            
            # Normalize by answer length (so short/long answers equal)
            answer_len = query_mask.sum(dim=1, keepdim=True).clamp_min(1)
            loss = loss / answer_len
            
            # Sum per sample, average across batch
            loss = loss.sum(dim=1).mean()
            
            # Gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            accum_loss += loss.detach().item()
            
            accelerator.backward(loss)
            accum_steps += 1
            
            # Update
            if accum_steps % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                # Log
                if completed % args.logging_steps == 0 and accelerator.is_main_process:
                    lr = scheduler.get_last_lr()[0]
                    tqdm.write(f"[{completed}/{args.num_training_steps}] loss={accum_loss:.4f} lr={lr:.2e}")
                
                # Evaluate
                if completed > 0 and completed % args.evaluation_interval == 0:
                    model.eval()
                    val_loss = 0.0
                    val_batches = 0
                    
                    for val_batch in eval_dl:
                        val_ids = val_batch["input_ids"].to(accelerator.device)
                        val_qm = val_batch["query_mask"].to(accelerator.device).float()
                        
                        vbsz, vseq = val_ids.shape
                        val_attn = torch.ones((vbsz, vseq), dtype=torch.long, device=accelerator.device)
                        
                        vt = torch.rand(vbsz, 1, device=accelerator.device).expand(vbsz, vseq).clamp_min(1e-5)
                        vmask = (torch.bernoulli(vt) * val_qm).bool()
                        
                        vmasked = val_ids.masked_fill(vmask, tokenizer.mask_token_id)
                        vlabels = val_ids.masked_fill(~vmask, -100)
                        
                        with torch.inference_mode():
                            vlogits = model(input_ids=vmasked, attention_mask=val_attn).logits
                        
                        vloss = loss_func(vlogits.view(vbsz * vseq, -1), vlabels.view(-1))
                        vloss = vloss.view(vbsz, vseq) / vt
                        vlen = val_qm.sum(dim=1, keepdim=True).clamp_min(1)
                        vloss = (vloss / vlen).sum(dim=1).mean()
                        
                        if accelerator.num_processes > 1:
                            vloss = accelerator.gather_for_metrics(vloss).mean()
                        
                        val_loss += vloss.item()
                        val_batches += 1
                    
                    val_loss /= max(val_batches, 1)
                    
                    if accelerator.is_main_process:
                        tqdm.write(f"[{completed}] val_loss={val_loss:.4f}")
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            tqdm.write(f"      New best! Saving...")
                            best_dir = os.path.join(exp_dir, "best_model")
                            os.makedirs(best_dir, exist_ok=True)
                            state = accelerator.get_state_dict(model)
                            save_file(state, os.path.join(best_dir, "model.safetensors"))
                    
                    model.train()
                
                # Checkpoint
                if completed > 0 and completed % args.checkpoint_interval == 0:
                    if accelerator.is_main_process:
                        ckpt_dir = os.path.join(exp_dir, f"checkpoint_{completed}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        state = accelerator.get_state_dict(model)
                        save_file(state, os.path.join(ckpt_dir, "model.safetensors"))
                        tqdm.write(f"Saved: {ckpt_dir}")
                
                completed += 1
                progress.update(1)
                accum_loss = 0.0
                
                if completed >= args.num_training_steps:
                    break
    
    # Save final
    if accelerator.is_main_process:
        final_dir = os.path.join(exp_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        state = accelerator.get_state_dict(model)
        save_file(state, os.path.join(final_dir, "model.safetensors"))
        tokenizer.save_pretrained(final_dir)
        
        accelerator.print("\n" + "=" * 60)
        accelerator.print("TRAINING COMPLETE!")
        accelerator.print("=" * 60)
        accelerator.print(f"Final: {final_dir}")
        accelerator.print(f"Best val_loss: {best_val_loss:.4f}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()