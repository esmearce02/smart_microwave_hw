"""
Standalone MLX90640 diagnostic — run on Pi to verify sensor reads.
  python test_thermal.py
"""

import time
import board
import busio
import adafruit_mlx90640

# ── I2C init (try 800 kHz per Adafruit docs, fall back to board.I2C) ─────────
try:
    i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
    print("[I2C] busio.I2C at 800 kHz  OK")
except Exception as e:
    print(f"[I2C] busio.I2C failed ({e}), trying board.I2C()")
    i2c = board.I2C()
    print("[I2C] board.I2C()  OK")

# ── Sensor init ───────────────────────────────────────────────────────────────
mlx = adafruit_mlx90640.MLX90640(i2c)
print(f"[MLX] Serial: {[hex(x) for x in mlx.serial_number]}")
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
print("[MLX] Refresh rate set to 2 Hz")

# ── Read 5 frames ─────────────────────────────────────────────────────────────
frame = [0.0] * 768
print("\nReading 5 frames (takes ~5 s) ...\n")

for i in range(5):
    stamp = time.monotonic()
    try:
        mlx.getFrame(frame)
    except ValueError:
        print(f"  Frame {i}: ValueError (transient) — retrying")
        continue

    elapsed = time.monotonic() - stamp

    # All pixel stats
    t_min  = min(frame)
    t_max  = max(frame)
    t_mean = sum(frame) / len(frame)

    # Centre 8×8 hotspot
    centre = [frame[r * 32 + c] for r in range(8, 16) for c in range(12, 20)]
    c_max  = max(centre)
    c_min  = min(centre)

    print(f"  Frame {i} ({elapsed:.2f}s)")
    print(f"    All pixels  — min: {t_min:.1f}°C  max: {t_max:.1f}°C  mean: {t_mean:.1f}°C")
    print(f"    Centre 8×8  — min: {c_min:.1f}°C  max: {c_max:.1f}°C"
          f"  →  {c_max*9/5+32:.1f}°F")
    print()

print("Done.")
