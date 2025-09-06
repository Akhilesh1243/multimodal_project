import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = Path("../data/hateful_memes")

def show_sample():
    with open(DATA_DIR / "train.jsonl") as f:
        line = f.readline()
        sample = json.loads(line)

    print("Text:", sample["text"])
    print("Label:", sample["label"])

    img_rel = Path(sample["img"])
    img_path = DATA_DIR / img_rel if img_rel.parent != Path('.') else DATA_DIR / "img" / img_rel.name

    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    show_sample()
