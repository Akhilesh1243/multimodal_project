import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from dataset import HatefulMemesDataset, collate_batch
from model import MultiModalClassifier

DATA_ROOT = Path("../data/hateful_memes")
TRAIN_JSONL = DATA_ROOT / "train.jsonl"

VAL_CANDIDATES = [DATA_ROOT / "dev.jsonl", DATA_ROOT / "val.jsonl"]
VAL_JSONL = next((p for p in VAL_CANDIDATES if p.exists()), VAL_CANDIDATES[0])


def make_loaders(batch_size=4, max_len=64, img_size=128, model_name="distilbert-base-uncased"):
    """Smaller defaults for a smooth first run on Windows/CPU."""
    tok = AutoTokenizer.from_pretrained(model_name)

    train_ds = HatefulMemesDataset(TRAIN_JSONL, tokenizer=tok, max_len=max_len, img_size=img_size)
    val_ds   = HatefulMemesDataset(VAL_JSONL,   tokenizer=tok, max_len=max_len, img_size=img_size)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, collate_fn=collate_batch)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, collate_fn=collate_batch)
    return train_dl, val_dl


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds, all_labels = [], []

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attn      = batch["attention_mask"].to(device)
        images    = batch["images"].to(device)
        labels    = batch["labels"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(input_ids=input_ids, attention_mask=attn, images=images)
            loss = loss_fn(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler: scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().tolist())

        if i % 10 == 0:
            print(f"{'Train' if train else 'Val'} batch {i}/{len(loader)}")

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def main():
    print(f"Using validation file: {VAL_JSONL.name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl = make_loaders()  # smaller defaults for first run

    model = MultiModalClassifier(num_classes=2).to(device)

    set_requires_grad(model.text_encoder, False)
    set_requires_grad(model.image_encoder, False)
    print("Epoch 1: encoders are FROZEN (training only the fusion head).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    epochs = 2
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps
    )

    best_f1 = 0.0
    for ep in range(1, epochs + 1):
        if ep == 2:
            set_requires_grad(model.text_encoder, True)
            set_requires_grad(model.image_encoder, True)
            print("Epoch 2: encoders are UNFROZEN (full fine-tuning).")

        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_dl, device, optimizer, scheduler)
        val_loss, val_acc, val_f1 = run_epoch(model, val_dl, device)

        print(f"Epoch {ep}:")
        print(f"  Train | loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f}")
        print(f"  Val   | loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            outdir = Path("../artifacts"); outdir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), outdir / "best_multimodal.pt")
            print(f"  Saved best model (F1={best_f1:.4f}).")

if __name__ == "__main__":
    main()
