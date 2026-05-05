"""
Standalone MLX90640 diagnostic — run on Pi to verify sensor reads.
  python test_thermal.py
"""

import time
import adafruit_mlx90640

# ── I2C init — use ExtendedI2C(1) to open /dev/i2c-1 directly by bus number.
# busio.I2C(board.SCL, board.SDA) fails on Pi 5 because blinka can't resolve
# the RP1 GPIO chip pins. ExtendedI2C bypasses that lookup entirely.
try:
    from adafruit_extended_bus import ExtendedI2C as I2C
    i2c = I2C(1)
    print("[I2C] ExtendedI2C bus 1 (/dev/i2c-1)  OK")
except Exception as e:
    raise SystemExit(
        f"[I2C] ExtendedI2C failed: {e}\n"
        "Run:  pip install adafruit-extended-bus"
    )

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
