import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import HatefulMemesDataset, collate_batch

TRAIN_JSONL = "../data/hateful_memes/train.jsonl"
MODEL_NAME = "distilbert-base-uncased"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    ds = HatefulMemesDataset(TRAIN_JSONL, tokenizer=tok, max_len=32, img_size=128)
    print("Dataset size:", len(ds))

    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_batch)

    batch = next(iter(dl))

    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    if "token_type_ids" in batch:
        print("token_type_ids shape:", batch["token_type_ids"].shape)
    print("images shape:", batch["images"].shape)
    print("labels shape:", batch["labels"].shape)

    print("Sample texts:", batch["texts"])

if __name__ == "__main__":
    main()
