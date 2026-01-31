import time
import streamlit as st
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


st.set_page_config(page_title="Telugu Diffusion Q/A", layout="wide")


@st.cache_resource
def load_model(repo_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForMaskedLM.from_pretrained(repo_id)
    model.to(device)
    model.eval()
    return model, tok


def prepare_conditional(tok, question_text: str, seq_len: int, device: str):
    """Prepare input matching training format."""
    chat = [{"role": "user", "content": question_text}]
    
    prompt_str = tok.apply_chat_template(
        chat,
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=True,
    )
    prompt_ids = tok.encode(prompt_str, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    
    # Full sequence like training: [prompt] + [MASK...] for answer region
    x = torch.full((1, seq_len), tok.mask_token_id, dtype=torch.long, device=device)
    x[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    
    # Mask = which positions can be modified (only answer region, not prompt)
    mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
    mask[0, :prompt_len] = False
    
    attn = torch.ones((1, seq_len), dtype=torch.long, device=device)
    
    return x, mask, attn, prompt_len


def decode_answer_region(tok, token_ids_1d, show_masks=False):
    """Decode tokens, stopping at EOS, showing masks optionally."""
    tokens = token_ids_1d.tolist()
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id
    mask_id = tok.mask_token_id
    
    result = []
    for t in tokens:
        if t == eos_id:
            break  # Stop at EOS
        elif t == pad_id:
            continue  # Skip PAD
        elif t == mask_id:
            if show_masks:
                result.append("▢")  # Show mask placeholder
        else:
            result.append(tok.decode([t]))
    
    return "".join(result)


def detect_repetition(tokens, window=4, threshold=3):
    """Detect if we're in a repetition loop."""
    if len(tokens) < window * threshold:
        return False, 0
    
    # Check for repeating patterns
    for pattern_len in range(1, window + 1):
        pattern = tokens[-pattern_len:]
        count = 0
        for i in range(2, threshold + 2):
            start = len(tokens) - (i * pattern_len)
            if start < 0:
                break
            if tokens[start:start + pattern_len] == pattern:
                count += 1
        if count >= threshold - 1:
            return True, len(tokens) - (threshold * pattern_len)
    return False, 0


@torch.no_grad()
def run_diffusion(model, tok, x, mask, attn, steps, temperature, strategy, prompt_len, 
                  repetition_penalty=1.5, max_answer_tokens=None):
    """
    Iterative denoising with:
    - EOS detection and handling
    - Repetition penalty
    - Repetition loop detection (early stop)
    - Optional max answer length
    """
    device = x.device
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id
    mask_id = tok.mask_token_id
    
    times = torch.linspace(1.0, 0.0, steps + 1, device=device).clamp_min(1e-5)
    
    eos_position = None
    generated_tokens = set()
    early_stop_pos = None

    for i, (t, s) in enumerate(zip(times[:-1], times[1:])):
        logits = model(input_ids=x, attention_mask=attn).logits
        
        # Apply repetition penalty
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            for token_id in generated_tokens:
                logits[0, prompt_len:, token_id] /= repetition_penalty
        
        # Boost EOS probability to encourage stopping
        logits[0, prompt_len:, eos_id] *= 1.2

        # Sample tokens for masked positions
        if mask.any():
            masked_logits = logits[mask] / max(temperature, 1e-6)
            probs = torch.softmax(masked_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            x[mask] = sampled
            generated_tokens.update(sampled.tolist())

        # Check for EOS
        answer_region = x[0, prompt_len:]
        eos_positions = (answer_region == eos_id).nonzero(as_tuple=True)[0]
        
        if len(eos_positions) > 0:
            first_eos = eos_positions[0].item()
            eos_position = prompt_len + first_eos
            x[0, eos_position + 1:] = pad_id
            mask[0, eos_position:] = False
        
        # Check for repetition loop
        answer_tokens = x[0, prompt_len:].tolist()
        # Filter out mask and pad tokens for repetition check
        real_tokens = [t for t in answer_tokens if t not in [mask_id, pad_id]]
        is_repeating, repeat_start = detect_repetition(real_tokens)
        
        if is_repeating and early_stop_pos is None:
            # Found repetition - truncate and stop
            early_stop_pos = prompt_len + repeat_start
            x[0, early_stop_pos:] = pad_id
            mask[0, early_stop_pos:] = False
        
        # Optional max answer length
        if max_answer_tokens and eos_position is None:
            current_len = (x[0, prompt_len:] != mask_id).sum().item()
            if current_len >= max_answer_tokens:
                x[0, prompt_len + max_answer_tokens:] = pad_id
                mask[0, prompt_len + max_answer_tokens:] = False

        # Remasking
        if mask.any():
            if strategy == "random":
                remask_probs = torch.rand_like(mask, dtype=torch.float, device=device) < (s / t)
                mask = mask & remask_probs
            else:
                probs_all = torch.softmax(logits, dim=-1)
                chosen_probs = torch.gather(probs_all, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
                chosen_probs[~mask] = 1.0

                num_to_remask = int((s / t) * mask.sum().item())
                if num_to_remask > 0:
                    masked_probs = chosen_probs.clone()
                    masked_probs[~mask] = float('inf')
                    flat_probs = masked_probs.view(-1)
                    k = min(num_to_remask, mask.sum().item())
                    _, lowest_indices = torch.topk(flat_probs, k=k, largest=False)
                    new_mask = torch.zeros_like(mask)
                    new_mask.view(-1)[lowest_indices] = True
                    mask = new_mask
            
            x[mask] = mask_id

        yield i + 1, x.clone(), mask.clone(), eos_position, early_stop_pos


# CSS
st.markdown("""
<style>
.output-box {
  font-size: 24px;
  line-height: 1.8;
  padding: 20px;
  border-radius: 12px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  min-height: 100px;
}
.final-answer {
  font-size: 26px;
  padding: 24px;
  border-radius: 12px;
  background: rgba(0,128,0,0.15);
  border: 2px solid rgba(0,128,0,0.4);
}
.mask-char { color: #666; }
</style>
""", unsafe_allow_html=True)


st.title("🌸 Telugu Diffusion LM")
st.caption("Iterative denoising for Telugu text generation")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    repo_id = st.text_input("Model", value="Prahaladha/telugu-diffusion-lm")
    device = st.selectbox("Device", ["cuda", "cpu"], 
                          index=0 if torch.cuda.is_available() else 1)

    st.subheader("Generation")
    seq_len = st.slider("Sequence length", 64, 256, 96, step=16,
                        help="Total sequence length (prompt + answer)")
    max_answer = st.slider("Max answer tokens", 8, 64, 24, step=4,
                           help="Stop after this many answer tokens")
    steps = st.slider("Diffusion steps", 10, 100, 40, step=5)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.5, step=0.1)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 3.0, 2.0, step=0.1,
                                   help="Higher = less repetition")
    
    st.subheader("Display")
    step_delay = st.slider("Step delay (sec)", 0.0, 0.15, 0.02, step=0.01)
    strategy = st.selectbox("Remasking", ["low_confidence", "random"])

# Input
st.subheader("Ask in Telugu")
question = st.text_input("Question:", value="భారతదేశ రాజధాని ఏమిటి?",
                         label_visibility="collapsed")

with st.expander("📝 Examples"):
    for ex in ["భారతదేశ రాజధాని ఏమిటి?", 
               "నీరు ఎందుకు ముఖ్యమైనది?",
               "సూర్యుడు ఎందుకు ప్రకాశిస్తాడు?"]:
        if st.button(ex, key=ex):
            question = ex

col1, col2 = st.columns(2)
with col1:
    go = st.button("▶️ Generate", type="primary", use_container_width=True)
with col2:
    stop = st.button("⏹️ Stop", use_container_width=True)

if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False
if stop:
    st.session_state.stop_flag = True

status = st.empty()
output = st.empty()
final = st.empty()

if go:
    st.session_state.stop_flag = False
    
    try:
        with st.spinner("Loading model..."):
            model, tok = load_model(repo_id, device)
        
        x, mask, attn, prompt_len = prepare_conditional(tok, question, seq_len, device)
        answer_len = seq_len - prompt_len
        
        status.info(f"Prompt: {prompt_len} tokens | Answer region: {answer_len} tokens")
        
        final_x = None
        final_eos = None
        final_early_stop = None
        
        for step_i, x, current_mask, eos_pos, early_stop in run_diffusion(
            model, tok, x, mask, attn, steps, temperature, strategy, prompt_len,
            repetition_penalty=repetition_penalty, max_answer_tokens=max_answer
        ):
            if st.session_state.stop_flag:
                status.warning("Stopped")
                break
            
            final_x = x.clone()
            final_eos = eos_pos
            final_early_stop = early_stop
            
            # Display answer region
            answer_tokens = x[0, prompt_len:]
            txt = decode_answer_region(tok, answer_tokens, show_masks=True)
            
            masks_left = int(current_mask.sum().item())
            info = ""
            if eos_pos:
                info = f" | EOS at {eos_pos - prompt_len}"
            elif early_stop:
                info = f" | Repetition stopped at {early_stop - prompt_len}"
            
            status.progress(step_i / steps, 
                           text=f"Step {step_i}/{steps} | Masks: {masks_left}{info}")
            output.markdown(f"<div class='output-box'>{txt}</div>", unsafe_allow_html=True)
            
            if step_delay > 0:
                time.sleep(step_delay)
        
        # Final output
        if final_x is not None:
            answer_tokens = final_x[0, prompt_len:]
            final_text = decode_answer_region(tok, answer_tokens, show_masks=False)
            
            status.success("✅ Complete!")
            final.markdown(f"### Answer\n<div class='final-answer'>{final_text}</div>", 
                          unsafe_allow_html=True)
            
            # Info messages
            if final_eos:
                st.caption(f"✓ EOS generated at position {final_eos - prompt_len}")
            elif final_early_stop:
                st.info(f"⚠️ Repetition detected - truncated at position {final_early_stop - prompt_len}")
            else:
                st.warning("⚠️ No EOS generated. Try: lower temperature, higher repetition penalty, or retrain model.")
    
    except Exception as e:
        st.error(str(e))
        import traceback
        st.code(traceback.format_exc())