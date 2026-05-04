"""
Food detection module using Ultralytics YOLO.
Camera priority: Raspberry Pi AI Camera (picamera2) → USB webcam → simulation.
"""

import random
import time
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

try:
    from picamera2 import Picamera2
    from picamera2.devices.imx500 import IMX500
    _PICAM_OK = True
except (ImportError, RuntimeError):
    _PICAM_OK = False

# Class names must match the order in food.yaml
FOOD_CLASS_IDS = set(range(14))

FOOD_META = {
    "soup":             {"icon": "🍲", "initial_temp_f": 55, "desc": "Soup"},
    "rice":             {"icon": "🍚", "initial_temp_f": 40, "desc": "Rice"},
    "broccoli":         {"icon": "🥦", "initial_temp_f": 33, "desc": "Broccoli"},
    "chicken nugget":   {"icon": "🍗", "initial_temp_f": 38, "desc": "Chicken Nuggets"},
    "baked potato":     {"icon": "🥔", "initial_temp_f": 42, "desc": "Baked Potato"},
    "mashed potatoes":  {"icon": "🥔", "initial_temp_f": 45, "desc": "Mashed Potatoes"},
    "pasta":            {"icon": "🍝", "initial_temp_f": 40, "desc": "Pasta"},
    "grilled salmon":   {"icon": "🐟", "initial_temp_f": 35, "desc": "Grilled Salmon"},
    "green beans":      {"icon": "🫘", "initial_temp_f": 33, "desc": "Green Beans"},
    "pizza":            {"icon": "🍕", "initial_temp_f": 42, "desc": "Pizza"},
    "mac and cheese":   {"icon": "🧀", "initial_temp_f": 40, "desc": "Mac and Cheese"},
    "baked beans":      {"icon": "🫘", "initial_temp_f": 38, "desc": "Baked Beans"},
    "lasagna":          {"icon": "🍝", "initial_temp_f": 42, "desc": "Lasagna"},
    "milk":             {"icon": "🥛", "initial_temp_f": 38, "desc": "Milk"},
}

_SIM_FOODS = list(FOOD_META.keys())

_CLASS_NAMES = [
    "broccoli", "chicken nugget", "rice", "green beans", "grilled salmon",
    "lasagna", "mac and cheese", "mashed potatoes", "milk", "pasta",
    "pizza", "soup", "baked beans", "baked potato",
]

DATA_DIR = Path(__file__).parent / "data"


def _build_class_image_index() -> dict[str, list[Path]]:
    """Map each food name → list of image paths where that class dominates."""
    index: dict[str, list[Path]] = {name: [] for name in _CLASS_NAMES}
    for split in ("train", "val"):
        labels_dir = DATA_DIR / split / "labels"
        images_dir = DATA_DIR / split / "images"
        if not labels_dir.exists():
            continue
        for lf in labels_dir.glob("*.txt"):
            classes = []
            try:
                for line in lf.read_text().splitlines():
                    parts = line.strip().split()
                    if parts:
                        classes.append(int(parts[0]))
            except Exception:
                continue
            if not classes:
                continue
            dominant = max(set(classes), key=classes.count)
            if dominant < len(_CLASS_NAMES):
                img = images_dir / (lf.stem + ".jpg")
                if img.exists():
                    index[_CLASS_NAMES[dominant]].append(img)
    return index


class FoodDetector:
    """
    Wraps YOLO detection with camera priority:
      1. Raspberry Pi AI Camera (picamera2)
      2. USB webcam (cv2)
      3. Simulation fallback
    """

    def __init__(self, model_name: str = "models/best.pt"):
        self.model      = None
        self.picam      = None   # Raspberry Pi AI Camera
        self.cap        = None   # USB webcam fallback
        self.sim_mode   = True
        self.force_sim  = False
        self._sim_frame = None
        self._class_images = _build_class_image_index()
        self._model_name   = model_name
        self._load()

    # ── Initialization ────────────────────────────────────────────────────────

    def _load(self):
        if not _YOLO_OK:
            return
        try:
            self.model = YOLO(self._model_name)
        except Exception as exc:
            print(f"[FoodDetector] YOLO load error: {exc} — simulation mode.")
            return

        # 1. Try Raspberry Pi AI Camera (IMX500)
        if _PICAM_OK:
            try:
                imx500 = IMX500()              # initialise IMX500 sensor (frame capture only)
                self.picam = Picamera2(imx500.camera_num)
                cfg = self.picam.create_preview_configuration(
                    main={"format": "RGB888", "size": (640, 480)}
                )
                self.picam.configure(cfg)
                self.picam.start()
                self.sim_mode = False
                print("[FoodDetector] Raspberry Pi AI Camera (IMX500) + YOLO active.")
                return
            except Exception as exc:
                print(f"[FoodDetector] Pi AI Camera error: {exc}")
                self.picam = None

        # 2. Fall back to USB webcam
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.sim_mode = False
            print("[FoodDetector] USB webcam + YOLO active.")
        else:
            self.cap = None
            print("[FoodDetector] No camera found — simulation mode.")

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(self, duration: float = 3.0) -> dict | None:
        if self.sim_mode or self.force_sim:
            return self._sim_scan(duration)
        return self._yolo_scan(duration)

    def grab_frame(self) -> np.ndarray | None:
        if self._sim_frame is not None:
            return self._sim_frame
        if self.picam:
            frame_rgb = self.picam.capture_array()
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            return frame if ok else None
        return self._synthetic_frame()

    def release(self):
        if self.picam:
            self.picam.stop()
        if self.cap:
            self.cap.release()

    def clear_sim_frame(self):
        self._sim_frame = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _yolo_scan(self, duration: float) -> dict | None:
        deadline = time.time() + duration
        votes: dict[str, int] = {}
        while time.time() < deadline:
            frame = self.grab_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            results = self.model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    name = self.model.names[int(box.cls[0])]
                    if name in FOOD_META:
                        votes[name] = votes.get(name, 0) + 1
            time.sleep(0.05)
        if not votes:
            return None
        best = max(votes, key=lambda k: votes[k])
        return {"name": best, **FOOD_META[best]}

    def _sim_scan(self, duration: float) -> dict | None:
        time.sleep(duration)
        name = random.choice(_SIM_FOODS)
        images = self._class_images.get(name, [])
        if images:
            self._sim_frame = cv2.imread(str(random.choice(images)))
        return {"name": name, **FOOD_META[name]}

    @staticmethod
    def _synthetic_frame() -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (20, 20, 35)
        cv2.circle(frame, (320, 260), 200, (40, 40, 60), 2)
        y = int((time.time() * 80) % 480)
        cv2.line(frame, (0, y), (640, y), (0, 212, 255), 1)
        cv2.putText(frame, "SCANNING...", (220, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 255), 2)
        return frame
