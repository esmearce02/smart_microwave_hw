"""
Thermal camera module.
Uses MLX90640 hardware on Raspberry Pi when available,
falls back to Newton's Law simulation otherwise.
"""

import math
import random
import time


try:
    import board
    import adafruit_mlx90640
    _MLX_OK = True
except (ImportError, NotImplementedError):
    _MLX_OK = False


class ThermalCamera:
    """
    Reads food temperature from an MLX90640 IR sensor (32×24 pixels, I2C).
    Falls back to a simulated heating curve when hardware is unavailable.

    Simulation uses Newton's Law of Cooling in reverse:
        T(t) = T_env - (T_env - T_0) * exp(-t / tau)
    """

    TARGET_F = 165.0
    ENV_F    = 212.0
    NOISE_SD = 0.4

    def __init__(self, initial_temp_f: float = 42.0, heat_seconds: float = 40.0):
        self._t0          = initial_temp_f
        self._start       = None
        self._tau         = self._calc_tau(heat_seconds)
        self._done        = False
        self._frozen_temp = None

        # MLX90640 hardware
        self._mlx         = None
        self._mlx_buf     = [0.0] * 768   # 32×24 pixel buffer
        self._last_mlx_t  = None          # cached reading (sensor runs at 4 Hz)
        self._last_mlx_ts = 0.0

        if _MLX_OK:
            try:
                i2c = board.I2C()
                self._mlx = adafruit_mlx90640.MLX90640(i2c)
                self._mlx.refresh_rate = adafruit_mlx90640.RefreshRate.RATE_4_HZ
                print("[ThermalCamera] MLX90640 active.")
            except Exception as exc:
                print(f"[ThermalCamera] MLX90640 error: {exc} — simulation mode.")
                self._mlx = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def hw_mode(self) -> bool:
        return self._mlx is not None

    def start(self):
        self._start       = time.time()
        self._done        = False
        self._frozen_temp = None

    def stop(self):
        """Freeze temperature at current reading."""
        self._frozen_temp = self.temperature

    def resume(self, heat_seconds_remaining: float = 20.0):
        """Continue heating from the frozen temperature."""
        if self._frozen_temp is not None:
            self._t0          = self._frozen_temp
            self._frozen_temp = None
            self._tau         = self._calc_tau(heat_seconds_remaining)
            self._start       = time.time()
            self._done        = False

    @property
    def temperature(self) -> float:
        if self._frozen_temp is not None:
            return self._frozen_temp
        if self._mlx:
            return self._read_mlx()
        return self._sim_temperature()

    @property
    def reached_target(self) -> bool:
        if self._mlx:
            return self.temperature >= self.TARGET_F
        return self._done

    def reset(self, initial_temp_f: float = 42.0, heat_seconds: float = 40.0):
        self._t0          = initial_temp_f
        self._start       = None
        self._tau         = self._calc_tau(heat_seconds)
        self._done        = False
        self._frozen_temp = None

    def colormap_frame(self, width: int = 400, height: int = 120):
        """Return a BGR false-colour thermal image for the UI."""
        import numpy as np
        import cv2

        if self._mlx:
            return self._mlx_colormap_frame(width, height)
        return self._sim_colormap_frame(width, height)

    # ── MLX90640 hardware ─────────────────────────────────────────────────────

    def _read_mlx(self) -> float:
        """Read centre-region max temperature from the sensor (cached at 4 Hz)."""
        now = time.time()
        if now - self._last_mlx_ts < 0.25:          # use cached value between sensor frames
            return self._last_mlx_t or self._t0
        try:
            self._mlx.getFrame(self._mlx_buf)
            # centre 8×8 region of the 32×24 grid → food hotspot
            centre = [
                self._mlx_buf[r * 32 + c]
                for r in range(8, 16)
                for c in range(12, 20)
            ]
            temp_c = max(centre)
            temp_f = temp_c * 9 / 5 + 32
            self._last_mlx_t  = temp_f
            self._last_mlx_ts = now
            return temp_f
        except Exception:
            return self._last_mlx_t or self._t0

    def _mlx_colormap_frame(self, width: int, height: int):
        import numpy as np
        import cv2

        try:
            self._mlx.getFrame(self._mlx_buf)
            arr = np.array(self._mlx_buf, dtype=np.float32).reshape(24, 32)
            lo, hi = arr.min(), arr.max()
            norm = ((arr - lo) / max(hi - lo, 1.0) * 255).astype(np.uint8)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            colored = cv2.resize(colored, (width, height))
        except Exception:
            colored = self._sim_colormap_frame(width, height)

        temp_f = self.temperature
        cv2.putText(colored, f"{temp_f:.1f} °F", (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self._draw_target_line(colored, width, height)
        return colored

    # ── Simulation ────────────────────────────────────────────────────────────

    def _sim_temperature(self) -> float:
        if self._start is None:
            return self._t0
        if self._done:
            return self.TARGET_F
        elapsed = time.time() - self._start
        temp    = self.ENV_F - (self.ENV_F - self._t0) * math.exp(-elapsed / self._tau)
        temp    = min(temp + random.gauss(0, self.NOISE_SD), self.ENV_F)
        if temp >= self.TARGET_F:
            self._done = True
            return self.TARGET_F
        return temp

    def _sim_colormap_frame(self, width: int, height: int):
        import numpy as np
        import cv2

        temp = self.temperature
        norm = max(0.0, min(1.0, (temp - 32.0) / (212.0 - 32.0)))
        base    = np.full((height, width), int(norm * 255), dtype=np.uint8)
        colored = cv2.applyColorMap(base, cv2.COLORMAP_JET)
        cv2.putText(colored, f"{temp:.1f} °F", (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self._draw_target_line(colored, width, height)
        return colored

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _calc_tau(self, heat_seconds: float) -> float:
        ratio = (self.TARGET_F - self._t0) / (self.ENV_F - self._t0)
        ratio = min(ratio, 0.9999)
        return -heat_seconds / math.log(1.0 - ratio)

    @staticmethod
    def _draw_target_line(img, width: int, height: int):
        import cv2
        y = int((1.0 - (ThermalCamera.TARGET_F - 32.0) / 180.0) * height)
        cv2.line(img, (0, y), (width, y), (255, 255, 255), 1)
        cv2.putText(img, "165°F target", (4, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
