"""
Transfer learning: freeze YOLO backbone (layers 0-9), train detection head only.
Best weights saved to models/best.pt
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

MODEL     = r"C:\Users\esmea\.claude\projects\exp3.pt"
DATA      = str(Path(__file__).parent / "data" / "food.yaml")
EPOCHS    = 50
IMGSZ     = 640
BATCH     = 8
FREEZE    = 10   # freeze backbone (layers 0-9), train head only
LR        = 1e-3 # higher LR is fine since only the head is updating
PATIENCE  = 15   # stop if val mAP50 doesn't improve for 15 consecutive epochs

MODELS_DIR = Path(__file__).parent / "models"

if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)

    model = YOLO(MODEL)

    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        freeze=FREEZE,
        lr0=LR,
        patience=PATIENCE,
        project="runs/detect",
        name="train",
        exist_ok=False,
    )

    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = MODELS_DIR / "best.pt"
    shutil.copy(best_src, best_dst)

    print("\nTraining complete.")
    print(f"Best weights saved to: {best_dst}")
