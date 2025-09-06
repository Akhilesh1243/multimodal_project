import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from model import MultiModalClassifier
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = Path("../data/hateful_memes")
CKPT = Path("../artifacts/best_multimodal.pt")

MAX_LEN = 64    
IMG_SIZE = 128  


tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

assert CKPT.exists(), f"Checkpoint not found: {CKPT}"
model = MultiModalClassifier(num_classes=2).to(DEVICE)
state = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

def _resolve_img_path(img_field: str) -> Path:
    """Handle both 'img/xxxx.png' and 'xxxx.png' cases."""
    p = Path(img_field)
    path = DATA_DIR / p if p.parent != Path('.') else DATA_DIR / "img" / p.name
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path.resolve()}")
    return path

def _get_sample(idx=0, split="train"):
    """Read one JSONL record by index from train/dev/val."""
    if split == "val":
        cand = DATA_DIR / "dev.jsonl"
        jsonl = cand if cand.exists() else (DATA_DIR / "val.jsonl")
    else:
        jsonl = DATA_DIR / f"{split}.jsonl"

    with jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"Index {idx} out of range for {jsonl.name}")

def predict_one(text: str = None, image_path: str | Path = None, sample_idx: int | None = None, split="train"):
    
    if sample_idx is not None:
        s = _get_sample(sample_idx, split=split)
        if text is None:
            text = s["text"]
        if image_path is None:
            image_path = _resolve_img_path(s["img"])

    if text is None or image_path is None:
        raise ValueError("Provide either (text and image_path) or a sample_idx.")

    enc = tok(text, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attn      = enc["attention_mask"].to(DEVICE)

    img = Image.open(Path(image_path)).convert("RGB")
    img = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn, images=img)
        probs_t = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs_t).item())
        probs = probs_t.cpu().tolist()

    return {"not_hateful": probs[0], "hateful": probs[1], "pred": pred, "text": text, "image": str(image_path)}

if __name__ == "__main__":
    print(predict_one(sample_idx=0, split="train"))

