import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import HatefulMemesDataset, collate_batch
from model import MultiModalClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path("../data/hateful_memes")
VAL_JSONL = (DATA_ROOT / "dev.jsonl") if (DATA_ROOT / "dev.jsonl").exists() else (DATA_ROOT / "val.jsonl")
CKPT = Path("../artifacts/best_multimodal.pt")
MODEL_NAME = "distilbert-base-uncased"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = HatefulMemesDataset(VAL_JSONL, tokenizer=tok, max_len=128, img_size=224)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_batch)

    model = MultiModalClassifier(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    all_y, all_p = [], []
    with torch.no_grad():
        for b in dl:
            ids = b["input_ids"].to(DEVICE)
            attn = b["attention_mask"].to(DEVICE)
            imgs = b["images"].to(DEVICE)
            y = b["labels"].cpu().tolist()
            logits = model(input_ids=ids, attention_mask=attn, images=imgs)
            p = logits.argmax(dim=1).cpu().tolist()
            all_y.extend(y); all_p.extend(p)

    print(confusion_matrix(all_y, all_p))
    print(classification_report(all_y, all_p, digits=4))

if __name__ == "__main__":
    main()
