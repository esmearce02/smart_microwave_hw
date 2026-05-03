"""
Convert fp-microwave-food.ndjson to YOLO dataset layout.
Downloads images from CDN and writes YOLO .txt label files.

Output:
    data/train/images/<file>
    data/train/labels/<stem>.txt
    data/val/images/<file>
    data/val/labels/<stem>.txt
"""

import json
import urllib.request
from pathlib import Path

NDJSON   = Path(r"C:\Users\esmea\.claude\projects\fp-microwave-food.ndjson")
DATA_DIR = Path(__file__).parent / "data"


def download(url: str, dest: Path):
    if dest.exists():
        return
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        print(f"  WARN: could not download {dest.name}: {exc}")


def main():
    for split in ("train", "val"):
        (DATA_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    train_count = val_count = skip_count = 0

    with NDJSON.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            record = json.loads(raw)

            if record.get("type") != "image":
                continue

            split = record.get("split")
            if split not in ("train", "val"):
                skip_count += 1
                continue

            fname  = record["file"]
            url    = record["url"]
            boxes  = record.get("annotations", {}).get("boxes", [])

            img_path   = DATA_DIR / split / "images" / fname
            label_path = DATA_DIR / split / "labels" / (Path(fname).stem + ".txt")

            # Download image
            download(url, img_path)

            # Write YOLO label file  (class cx cy w h — already normalised)
            with label_path.open("w") as lf:
                for box in boxes:
                    cls, cx, cy, w, h = box
                    lf.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            if split == "train":
                train_count += 1
            else:
                val_count += 1

            if (train_count + val_count) % 100 == 0:
                print(f"  Processed {train_count + val_count} images…")

    print(f"\nDone. train={train_count}  val={val_count}  skipped={skip_count}")


if __name__ == "__main__":
    main()
