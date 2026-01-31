import argparse
import time
from datasets import load_dataset
from tokenizer import get_tokenizer


def parse_args():
    p = argparse.ArgumentParser("Prepare Telugu pretrain data (fixed-length blocks)")
    p.add_argument("--hf_model_name", type=str, default="ai4bharat/IndicBERTv2-MLM-only")
    p.add_argument("--wiki_name", type=str, default="wikimedia/wikipedia")
    p.add_argument("--wiki_config", type=str, default="20231101.te")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--test_split_pct", type=float, default=0.005)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_docs", type=int, default=0, help="0 = use all docs (can be huge)")
    p.add_argument("--path_to_data_store", type=str, required=True)
    p.add_argument("--huggingface_cache_dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    tok = get_tokenizer(args.hf_model_name)

    ds = load_dataset(
        args.wiki_name,
        args.wiki_config,
        split=args.split,
        cache_dir=args.huggingface_cache_dir,
    )

    if args.max_docs and args.max_docs > 0:
        ds = ds.select(range(min(args.max_docs, len(ds))))

    # keep only text column
    keep_col = "text"
    drop_cols = [c for c in ds.column_names if c != keep_col]
    ds = ds.remove_columns(drop_cols)

    ds = ds.train_test_split(test_size=args.test_split_pct, seed=42)

    def tokenize_and_chunk(examples):
        tokenized = tok(
            examples["text"],
            return_attention_mask=False,
            add_special_tokens=True,   # uses TemplateProcessing => BOS..EOS
            truncation=False,
            max_length=None,
        )

        blocks = []
        for ids in tokenized["input_ids"]:
            for i in range(0, len(ids), args.context_length):
                chunk = ids[i : i + args.context_length]
                if len(chunk) < args.context_length:
                    chunk = chunk + [tok.pad_token_id] * (args.context_length - len(chunk))
                blocks.append(chunk)
        return {"input_ids": blocks}

    out = ds.map(
        tokenize_and_chunk,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_workers,
        remove_columns=["text"],
    )

    print("Saving to:", args.path_to_data_store)
    out.save_to_disk(args.path_to_data_store)

    # quick load test
    start = time.time()
    _ = out["train"][0]["input_ids"]
    print("Example len:", len(_))
    print("Done. (prep time sanity check)", time.time() - start)


if __name__ == "__main__":
    main()
