import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HatefulMemesDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=128, img_size=224):
        self.jsonl_path = Path(jsonl_path)
        self.data_dir = self.jsonl_path.parent
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]
        self.tok = tokenizer
        self.max_len = max_len
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def _resolve_img_path(self, img_field):
        p = Path(img_field)
        if p.parent != Path('.'):
            return self.data_dir / p
        return self.data_dir / "img" / p.name

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = s["text"]
        label = int(s["label"])

        enc = self.tok(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        img_path = self._resolve_img_path(s["img"])
        image = Image.open(img_path).convert("RGB")
        image = self.tf(image)

        return {"text": text, "inputs": enc, "image": image, "label": label}


def collate_batch(batch):
    import torch
    input_ids = torch.stack([b["inputs"]["input_ids"] for b in batch])
    attention_mask = torch.stack([b["inputs"]["attention_mask"] for b in batch])
    
    if "token_type_ids" in batch[0]["inputs"]:
        token_type_ids = torch.stack([b["inputs"]["token_type_ids"] for b in batch])
    else:
        token_type_ids = None

    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    texts = [b["text"] for b in batch]

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": images,
        "labels": labels,
        "texts": texts
    }
    if token_type_ids is not None:
        out["token_type_ids"] = token_type_ids
    return out
