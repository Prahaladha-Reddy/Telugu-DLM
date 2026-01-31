import argparse
import torch
from safetensors.torch import load_file
from transformers import AutoModelForMaskedLM
from tokenizer import get_tokenizer


def load_model_and_tokenizer(weights_path: str, hf_model_name: str, device: str):
    tok = get_tokenizer(hf_model_name)
    model = AutoModelForMaskedLM.from_pretrained(hf_model_name)
    model.resize_token_embeddings(len(tok))
    state = load_file(weights_path)
    model.load_state_dict(state, strict=False)
    model.tie_weights()
    model.to(device)
    model.eval()
    return model, tok


def prepare_conditional(seq_len: int, tok, question_text: str, device: str):
    # Build prompt tokens using chat_template (no extra special tokens wrapper)
    chat = [{"role": "user", "content": question_text}]
    prompt_ids = tok.apply_chat_template(
        chat,
        tokenize=True,
        add_special_tokens=False,
        add_generation_prompt=True,
    )
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    x = torch.full((1, seq_len), tok.mask_token_id, dtype=torch.long, device=device)
    mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
    attn = torch.ones((1, seq_len), dtype=torch.long, device=device)

    n = min(prompt.shape[1], seq_len)
    x[:, :n] = prompt[:, :n]
    mask[:, :n] = False  # freeze prompt tokens
    return x, mask, attn, n


def decode_with_masks(tok, token_ids_1d):
    toks = tok.convert_ids_to_tokens(token_ids_1d.tolist())

    cleaned = []
    for t in toks:
        if t == tok.mask_token:
            cleaned.append(t)
        elif t in tok.all_special_tokens:
            continue
        else:
            cleaned.append(t)

    return tok.convert_tokens_to_string(cleaned)


@torch.no_grad()
def diffusion_generate(
    model,
    tok,
    x,
    mask,
    attn,
    steps: int,
    temperature: float = 1.0,
    strategy: str = "low_confidence",   # "random" or "low_confidence"
):
    device = x.device
    times = torch.linspace(1.0, 0.0, steps + 1, device=device).clamp_min(1e-5)

    for i, (t, s) in enumerate(zip(times[:-1], times[1:]), start=1):
        logits = model(input_ids=x, attention_mask=attn).logits

        # sample only currently masked positions
        if mask.any():
            probs = torch.softmax(logits[mask] / max(temperature, 1e-6), dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            x[mask] = sampled

        # remasking
        if strategy == "random":
            remask_probs = (torch.rand_like(mask, dtype=torch.float, device=device) < (s / t))
            mask = mask & remask_probs
            x[mask] = tok.mask_token_id

        elif strategy == "low_confidence":
            probs_all = torch.softmax(logits, dim=-1)
            chosen = torch.gather(probs_all, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
            chosen[~mask] = 1.0  # don't remask frozen/unmasked tokens

            num_to_remask = int((s / t) * mask.sum().item())
            if num_to_remask > 0:
                lowest_idx = torch.topk(chosen, k=num_to_remask, largest=False).indices
                new_mask = torch.zeros_like(mask)
                new_mask[0, lowest_idx[0]] = True
                mask = new_mask
                x[mask] = tok.mask_token_id

        yield i, x, mask


def main():
    ap = argparse.ArgumentParser("Telugu LDM Inference (CLI)")
    ap.add_argument("--weights", required=True, type=str)
    ap.add_argument("--hf_model_name", default="ai4bharat/IndicBERTv2-MLM-only", type=str)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--seq_len", default=512, type=int)
    ap.add_argument("--steps", default=64, type=int)
    ap.add_argument("--temperature", default=1.0, type=float)
    ap.add_argument("--strategy", default="low_confidence", choices=["random", "low_confidence"])
    ap.add_argument("--question", required=True, type=str)
    ap.add_argument("--print_every", default=1, type=int)
    args = ap.parse_args()

    model, tok = load_model_and_tokenizer(args.weights, args.hf_model_name, args.device)
    x, mask, attn, prompt_len = prepare_conditional(args.seq_len, tok, args.question, args.device)

    final_text = ""
    for step, x, mask in diffusion_generate(model, tok, x, mask, attn, args.steps, args.temperature, args.strategy):
        if step % args.print_every == 0:
            # show only generated area (after prompt tokens) to avoid clutter
            gen_part = x[0, prompt_len:]
            final_text = decode_with_masks(tok, gen_part)
            print(f"\nStep {step}/{args.steps}\n{final_text}\n")

    print("\n=== FINAL ===\n", final_text)


if __name__ == "__main__":
    main()
