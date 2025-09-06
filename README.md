# Multimodal Hateful Memes Classifier (DistilBERT + ResNet50)

End-to-end PyTorch project that classifies memes as **hateful (1)** or **not hateful (0)** by combining **text** and **image** signals.  
Text is encoded with **DistilBERT**, images with **ResNet50**, and a small **fusion MLP** makes the final prediction.

---

## Highlights
- **Multimodal**: Text (caption) + Image (meme) → fused prediction  
- **Pretrained encoders**: DistilBERT (768-dim) + ResNet50 (2048-dim)  
- **Clean pipeline**: dataset, DataLoader, training, evaluation, and inference scripts  
- **Beginner-friendly theory**: tokens, attention masks, tensors, fusion, softmax  
- **Repro tips**: seeds, sizes, checkpoint path, tokenizer name

---

## Repo Structure
multimodal_project/
├─ src/
│ ├─ dataset.py # JSONL + img loader, transforms, collate
│ ├─ model.py # DistilBERT + ResNet50 + fusion MLP
│ ├─ train.py # training loop (AdamW, scheduler, clip)
│ ├─ evaluate_saved.py # confusion matrix & classification report
│ └─ inference.py # single-sample prediction (demo API)
├─ data/
│ └─ hateful_memes/ # place dataset here (jsonl + img/)
├─ artifacts/ # saved checkpoints (.pt)
├─ requirements.txt
└─ .gitignore


---

## Setup (Windows / PowerShell)
```powershell
# from project root
python -m venv ml_env
.\ml_env\Scripts\activate
pip install -r requirements.txt

Data

Place files as:

data/hateful_memes/
  ├─ img/             # images
  ├─ train.jsonl
  ├─ dev.jsonl   (or val.jsonl)
  └─ test.jsonl  (optional for your experiments)

{"img": "img/42953.png", "text": "its their character not their color that matters", "label": 0}

Model Architecture (Text + Image → Fusion → Logits)

Text (caption) ──► DistilBERT ──► [CLS] 768-d ─┐
                                                 ├─► Concat (2816-d) ─► Linear(2816→512) ─► ReLU ─► Dropout ─► Linear(512→2) ─► logits
Image (meme)  ──► ResNet50  ───► GAP 2048-d ────┘


DistilBERT outputs a 768-dim sentence vector (using the first token position as summary).

ResNet50 (final fc → Identity) outputs a 2048-dim image vector.

We concatenate → MLP → 2 logits (not hateful / hateful).

Training uses CrossEntropyLoss on logits (softmax is implicit in the loss).



### Beginner Theory (short & clear)

Tokens: tiny chunks of text (words/subwords/punctuation). Tokenizer turns them into IDs.

Attention mask: marks real tokens vs padding (1 = keep, 0 = ignore).

Tensors: containers for numbers (vectors/matrices/higher-dim arrays).

DistilBERT (text): builds context-aware vectors via self-attention → 768-dim summary.

ResNet50 (image): CNN features → global average pool → 2048-dim vector.

Fusion: concat (768+2048=2816), MLP with ReLU & Dropout, then logits → softmax probabilities.

Training: Cross-Entropy + AdamW optimizer + linear scheduler + gradient clipping.

Inference: no gradients; forward pass, softmax, choose class (or apply a tuned threshold).

Shapes cheat-sheet (batch = B, tokens = L)

Text in: (B, L) IDs + mask → DistilBERT → (B, L, 768) → take index 0 → (B, 768)

Image in: (B, 3, H, W) → ResNet50 → (B, 2048)

Fusion: concat → (B, 2816) → MLP → logits (B, 2)


## Current Results (baseline)

Val accuracy: 0.566

Val macro-F1: 0.502

Confusion matrix shows low recall for hateful (class 1).