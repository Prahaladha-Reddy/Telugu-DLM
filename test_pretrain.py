import os
import random
import traceback
import streamlit as st
import torch

# If your transformers import is broken, FIX IT first (see terminal commands above).
from transformers import AutoModelForMaskedLM

from tokenizer import get_tokenizer


st.set_page_config(page_title="Telugu MLM Reconstruction Tester", layout="wide")


def normalize_local_path(p: str) -> str:
    p = (p or "").strip()
    if p and not p.startswith("/") and p.startswith("teamspace/"):
        p = "/" + p
    return os.path.expanduser(p)


@st.cache_resource
def load_model(checkpoint_dir: str, hf_model_name: str, device: str):
    checkpoint_dir = normalize_local_path(checkpoint_dir)
    tok = get_tokenizer(hf_model_name)

    if os.path.isdir(checkpoint_dir):
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_dir, local_files_only=True)
    else:
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_dir)

    model.to(device)
    model.eval()
    return model, tok


def mask_sentence(tok, text: str, mask_prob: float, max_len: int, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    enc = tok(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"][0]  # [L]

    special_ids = set(getattr(tok, "all_special_ids", []))
    mask_token_id = tok.mask_token_id

    # choose mask positions (avoid special tokens)
    positions = []
    for i, tid in enumerate(input_ids.tolist()):
        if tid in special_ids:
            continue
        if random.random() < mask_prob:
            positions.append(i)

    # guarantee at least 1 masked token
    if len(positions) == 0:
        for i, tid in enumerate(input_ids.tolist()):
            if tid not in special_ids:
                positions = [i]
                break

    masked_ids = input_ids.clone()
    masked_ids[positions] = mask_token_id

    return input_ids, masked_ids, positions


@torch.no_grad()
def reconstruct(model, tok, original_ids, masked_ids, positions, device: str, temperature: float, sampling: bool):
    original_ids = original_ids.to(device).unsqueeze(0)  # [1, L]
    masked_ids = masked_ids.to(device).unsqueeze(0)      # [1, L]

    attn = torch.ones_like(masked_ids, dtype=torch.long, device=device)

    logits = model(input_ids=masked_ids, attention_mask=attn).logits  # [1, L, V]

    out = masked_ids.clone()

    for pos in positions:
        dist = logits[0, pos] / max(float(temperature), 1e-6)
        if sampling:
            probs = torch.softmax(dist, dim=-1)
            pred = torch.multinomial(probs, num_samples=1).item()
        else:
            pred = torch.argmax(dist).item()
        out[0, pos] = pred

    # masked-token accuracy
    correct = 0
    for pos in positions:
        if out[0, pos].item() == original_ids[0, pos].item():
            correct += 1
    acc = correct / max(len(positions), 1)

    return out[0].cpu(), acc


def decode(tok, ids_1d):
    return tok.decode(ids_1d.tolist(), skip_special_tokens=True)


st.title("Telugu MLM Reconstruction Tester (mask → reconstruct)")

with st.sidebar:
    st.header("Settings")
    checkpoint_dir = st.text_input(
        "Checkpoint directory",
        value="/teamspace/studios/this_studio/runs/telugu_ldm_pretrain/final_model",
    )
    hf_model_name = st.text_input(
        "HF base MLM model (for tokenizer)",
        value="ai4bharat/IndicBERTv2-MLM-only",
    )
    device = st.selectbox("Device", ["cuda", "cpu"], index=0 if torch.cuda.is_available() else 1)

    max_len = st.slider("Max sequence length", 32, 512, 128, step=32)
    mask_prob = st.slider("Mask probability", 0.05, 0.80, 0.30, step=0.05)

    sampling = st.checkbox("Sampling (instead of argmax)", value=False)
    temperature = st.slider("Temperature", 0.2, 2.0, 1.0, step=0.1)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

examples = [
    "తెలుగు భాష భారతదేశంలో చాలా మంది మాట్లాడతారు.",
    "హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని.",
    "భారతదేశ రాజధాని న్యూఢిల్లీ.",
    "నేడు వాతావరణం చల్లగా ఉంది.",
]

text = st.text_area("Input sentence (Telugu):", value=examples[0], height=120)
go = st.button("▶ Mask & Reconstruct")

if go:
    try:
        model, tok = load_model(checkpoint_dir, hf_model_name, device)

        original_ids, masked_ids, positions = mask_sentence(
            tok=tok,
            text=text,
            mask_prob=float(mask_prob),
            max_len=int(max_len),
            seed=int(seed),
        )

        recon_ids, acc = reconstruct(
            model=model,
            tok=tok,
            original_ids=original_ids,
            masked_ids=masked_ids,
            positions=positions,
            device=device,
            temperature=float(temperature),
            sampling=bool(sampling),
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.write(decode(tok, original_ids))
            st.subheader("Masked input")
            st.write(decode(tok, masked_ids))
        with col2:
            st.subheader("Reconstructed")
            st.write(decode(tok, recon_ids))
            st.subheader("Masked-token accuracy")
            st.write(f"{acc*100:.2f}%  (masked positions: {len(positions)})")

        # Optional: show token-level info
        with st.expander("Show masked positions + tokens"):
            st.write("Positions:", positions)

    except Exception:
        st.error("CRASH (traceback below):")
        st.code(traceback.format_exc())
