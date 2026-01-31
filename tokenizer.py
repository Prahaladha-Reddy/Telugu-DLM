from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer


def get_tokenizer(
    model_name: str = "ai4bharat/IndicBERTv2-MLM-only",
    bos_token: str = "<BOS>",
    eos_token: str = "<EOS>",
    start_token: str = "<START_ID>",
    end_token: str = "<END_ID>",
    eot_token: str = "<EOT_ID>",
):
    """
    Adds repo-style special tokens + chat_template to any MLM tokenizer.

    IMPORTANT DESIGN CHOICE (to avoid double BOS/EOS):
    - We keep TemplateProcessing (BOS ... EOS) enabled for normal tokenization (pretrain script).
    - For chat_template tokenization (SFT + inference prompt), we will call apply_chat_template(add_special_tokens=False)
      so TemplateProcessing does NOT wrap again.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    special_tokens = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "additional_special_tokens": [start_token, end_token, eot_token],
    }
    tokenizer.add_special_tokens(special_tokens)

    # PAD/CLS choices like the reference repo
    tokenizer.pad_token = eos_token
    tokenizer.cls_token = bos_token

    # Pretraining post-processor: <BOS> ... <EOS>
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        special_tokens=[
            (bos_token, tokenizer.bos_token_id),
            (eos_token, tokenizer.eos_token_id),
        ],
    )

    # Chat template (same structure as repo)
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
