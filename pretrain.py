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

from tokenizer import get_tokenizer


def collate_fn(batch):
    tokens = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch])
    return {"input_ids": tokens}

stack_collate_fn = collate_fn


def parse_args():
    p = argparse.ArgumentParser("Telugu LDM Pretrain (MLM diffusion)")
    p.add_argument("--experiment_name", required=True, type=str)
    p.add_argument("--working_directory", required=True, type=str)

    p.add_argument("--hf_model_name", default="ai4bharat/IndicBERTv2-MLM-only", type=str)
    p.add_argument("--path_to_prepped_data", required=True, type=str)

    p.add_argument("--per_gpu_batch_size", default=16, type=int)
    p.add_argument("--gradient_accumulation_steps", default=1, type=int)
    p.add_argument("--num_training_steps", default=30000, type=int)

    p.add_argument("--max_grad_norm", default=1.0, type=float)
    p.add_argument("--lr_scheduler_type", default="cosine", type=str)
    p.add_argument("--num_warmup_steps", default=1000, type=int)
    p.add_argument("--learning_rate", default=5e-5, type=float)
    p.add_argument("--weight_decay", default=0.01, type=float)

    p.add_argument("--logging_steps", default=25, type=int)
    p.add_argument("--evaluation_interval", default=2000, type=int)
    p.add_argument("--checkpoint_interval", default=5000, type=int)

    p.add_argument("--log_wandb", default=False, action=argparse.BooleanOptionalAction)
    return p.parse_args()


def save_hf_checkpoint(accelerator: Accelerator, model, tok, save_dir: str, step: int | None = None):
    """
    Saves model + tokenizer safely even when weights are tied/shared.
    This avoids safetensors.save_file(shared_tensors_error).
    """
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(save_dir, safe_serialization=True)
        tok.save_pretrained(save_dir)

        if step is not None:
            with open(os.path.join(save_dir, "step.txt"), "w", encoding="utf-8") as f:
                f.write(str(step))

    accelerator.wait_for_everyone()


def evaluate(model, dataloader, tok, accelerator, loss_func):
    model.eval()
    total = 0.0
    n = 0
    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(accelerator.device)
            bsz, seq_len = input_ids.shape
            attention_mask = torch.ones((bsz, seq_len), dtype=torch.long, device=accelerator.device)

            t = torch.rand(bsz, 1, device=accelerator.device).expand(bsz, seq_len).clamp_min(1e-5)
            mask = torch.bernoulli(t).bool()

            masked_input_ids = input_ids.masked_fill(mask, tok.mask_token_id)
            labels = input_ids.masked_fill(~mask, -100)

            logits = model(input_ids=masked_input_ids, attention_mask=attention_mask).logits
            num_classes = logits.shape[-1]
            loss = loss_func(logits.reshape(bsz * seq_len, num_classes), labels.flatten())
            loss = (loss.reshape(bsz, seq_len) / t).mean()

            loss = loss.detach()
            if accelerator.num_processes > 1:
                loss = torch.mean(accelerator.gather_for_metrics(loss))

            total += loss.item()
            n += 1

    model.train()
    return total / max(n, 1)


def main():
    args = parse_args()
    path_to_experiment = os.path.join(args.working_directory, args.experiment_name)

    accelerator = Accelerator(
        project_dir=path_to_experiment,
        log_with="wandb" if args.log_wandb else None,
        mixed_precision="bf16",
    )
    if args.log_wandb:
        accelerator.init_trackers(args.experiment_name)

    tok = get_tokenizer(args.hf_model_name)

    model = AutoModelForMaskedLM.from_pretrained(args.hf_model_name)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.resize_token_embeddings(len(tok))
    model.tie_weights()  # this creates shared tensors -> must NOT use safetensors.save_file(state_dict)

    params = sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)
    accelerator.print("Number of Parameters:", params)

    mini_bs = args.per_gpu_batch_size // args.gradient_accumulation_steps

    data = load_from_disk(args.path_to_prepped_data)
    train_dl = DataLoader(data["train"], batch_size=mini_bs, shuffle=True, collate_fn=stack_collate_fn)
    eval_dl = DataLoader(data["test"], batch_size=mini_bs, shuffle=False, collate_fn=stack_collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
    )

    loss_func = nn.CrossEntropyLoss(reduction="none")

    model, optimizer, train_dl, eval_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, eval_dl, scheduler
    )

    completed = 0
    progress = tqdm(range(args.num_training_steps), disable=not accelerator.is_local_main_process)

    model.train()
    accum_steps = 0
    accum_loss = 0.0

    while completed < args.num_training_steps:
        for batch in train_dl:
            input_ids = batch["input_ids"].to(accelerator.device)
            bsz, seq_len = input_ids.shape
            attention_mask = torch.ones((bsz, seq_len), dtype=torch.long, device=accelerator.device)

            t = torch.rand(bsz, 1, device=accelerator.device).expand(bsz, seq_len).clamp_min(1e-5)
            mask = torch.bernoulli(t).bool()

            masked_input_ids = input_ids.masked_fill(mask, tok.mask_token_id)
            labels = input_ids.masked_fill(~mask, -100)

            logits = model(input_ids=masked_input_ids, attention_mask=attention_mask).logits
            num_classes = logits.shape[-1]
            loss = loss_func(logits.reshape(bsz * seq_len, num_classes), labels.flatten())
            loss = (loss.reshape(bsz, seq_len) / t).mean()

            loss = loss / args.gradient_accumulation_steps
            accum_loss += loss.detach().float().item()

            accelerator.backward(loss)
            accum_steps += 1

            if accum_steps % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if completed % args.logging_steps == 0 and accelerator.is_main_process:
                    tqdm.write(
                        f"[{completed}/{args.num_training_steps}] train_loss={accum_loss:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

                if completed % args.evaluation_interval == 0 and completed > 0:
                    val = evaluate(model, eval_dl, tok, accelerator, loss_func)
                    if accelerator.is_main_process:
                        tqdm.write(f"[{completed}] val_loss={val:.4f}")

                # ✅ FIXED: checkpoint saving (no safetensors.save_file(state_dict))
                if completed % args.checkpoint_interval == 0 and completed > 0:
                    ckpt_path = os.path.join(path_to_experiment, f"checkpoint_{completed}")
                    save_hf_checkpoint(accelerator, model, tok, ckpt_path, step=completed)
                    if accelerator.is_main_process:
                        tqdm.write(f"Saved checkpoint: {ckpt_path}")

                completed += 1
                progress.update(1)
                accum_loss = 0.0

                if completed >= args.num_training_steps:
                    break

    # ✅ FIXED: final save uses the SAME accelerator (do NOT create a new one)
    final_dir = os.path.join(path_to_experiment, "final_model")
    save_hf_checkpoint(accelerator, model, tok, final_dir, step=completed)
    if accelerator.is_main_process:
        tqdm.write(f"Saved final: {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
