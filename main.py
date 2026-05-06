"""
Smart Microwave – Food Recognition Simulation
============================================
State machine:
  IDLE  →  SCANNING  →  CONFIRMED  →  HEATING  →  COMPLETE
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import math
import platform
import cv2
import numpy as np
from PIL import Image, ImageTk

from detector import FoodDetector
from thermal  import ThermalCamera

# ── Palette ────────────────────────────────────────────────────────────────────
C = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "display":   "#0a1628",
    "border":    "#30363d",
    "accent":    "#00d4ff",
    "green":     "#3fb950",
    "amber":     "#d29922",
    "red":       "#f85149",
    "white":     "#e6edf3",
    "dim":       "#6e7681",
    "btn":       "#21262d",
    "btn_hover": "#30363d",
}

# ── States ─────────────────────────────────────────────────────────────────────
IDLE      = "IDLE"
SCANNING  = "SCANNING"
CONFIRMED = "CONFIRMED"
HEATING   = "HEATING"
COMPLETE  = "COMPLETE"


class SmartMicrowaveApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Microwave – Food Recognition System")
        self.configure(bg=C["bg"])
        self.resizable(False, False)

        self.detector = FoodDetector("models/best.pt")
        self.thermal  = ThermalCamera()

        self.state             = IDLE
        self._food             = None
        self._heat_seconds     = 40.0
        self._running          = True
        self._sim_clear_timer  = None  # after() ID for 10s sim-frame clear

        self._build_ui()
        self._enter_idle()
        self._tick()                   # start GUI refresh loop
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════════════
    # UI Construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        pad = dict(padx=10, pady=8)

        # ── Title bar ─────────────────────────────────────────────────────────
        title_frame = tk.Frame(self, bg=C["bg"])
        title_frame.pack(fill="x", **pad)
        tk.Label(title_frame, text="⚡ SMART MICROWAVE", font=("Courier New", 16, "bold"),
                 fg=C["accent"], bg=C["bg"]).pack(side="left")
        self._mode_lbl = tk.Label(title_frame, text="[ SIM ]",
                                  font=("Courier New", 10), fg=C["amber"], bg=C["bg"])
        self._mode_lbl.pack(side="right")

        # ── Main row: camera | controls ───────────────────────────────────────
        row = tk.Frame(self, bg=C["bg"])
        row.pack(fill="both", expand=True, padx=10)

        self._build_camera_panel(row)
        self._build_control_panel(row)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Place food inside and press START.")
        status = tk.Label(self, textvariable=self._status_var,
                          font=("Courier New", 10), fg=C["dim"], bg=C["bg"],
                          anchor="w", wraplength=780)
        status.pack(fill="x", padx=12, pady=(0, 8))

    # ── Camera panel ──────────────────────────────────────────────────────────

    def _build_camera_panel(self, parent):
        frame = tk.Frame(parent, bg=C["panel"], bd=1, relief="flat",
                         highlightbackground=C["border"], highlightthickness=1)
        frame.pack(side="left", padx=(0, 8), pady=4)

        tk.Label(frame, text="CAMERA FEED", font=("Courier New", 9, "bold"),
                 fg=C["dim"], bg=C["panel"]).pack(pady=(6, 2))

        self._cam_canvas = tk.Canvas(frame, width=400, height=300,
                                     bg="#050a10", highlightthickness=0)
        self._cam_canvas.pack(padx=8, pady=(0, 4))

        tk.Label(frame, text="── THERMAL IR ──", font=("Courier New", 8),
                 fg=C["dim"], bg=C["panel"]).pack(pady=(2, 2))

        self._ir_canvas = tk.Canvas(frame, width=400, height=120,
                                    bg="#050a10", highlightthickness=0)
        self._ir_canvas.pack(padx=8, pady=(0, 8))

    # ── Control panel ─────────────────────────────────────────────────────────

    def _build_control_panel(self, parent):
        frame = tk.Frame(parent, bg=C["panel"], bd=1, relief="flat",
                         highlightbackground=C["border"], highlightthickness=1)
        frame.pack(side="left", fill="both", expand=True, pady=4)

        # LCD display
        disp = tk.Frame(frame, bg=C["display"], bd=0,
                        highlightbackground=C["accent"], highlightthickness=1)
        disp.pack(fill="x", padx=12, pady=(12, 8))

        self._food_lbl = tk.Label(disp, text="---", font=("Courier New", 26, "bold"),
                                  fg=C["green"], bg=C["display"])
        self._food_lbl.pack(pady=(8, 2))

        self._food_desc = tk.Label(disp, text="Insert food & press START",
                                   font=("Courier New", 10), fg=C["dim"], bg=C["display"])
        self._food_desc.pack(pady=(0, 6))

        self._temp_var = tk.StringVar(value="-- °F")
        tk.Label(disp, textvariable=self._temp_var,
                 font=("Courier New", 20, "bold"), fg=C["amber"], bg=C["display"]
                 ).pack(pady=(0, 8))

        # Progress bar
        self._progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Heat.Horizontal.TProgressbar",
                        troughcolor=C["display"], background=C["red"],
                        lightcolor=C["red"], darkcolor=C["red"], borderwidth=0)
        pb = ttk.Progressbar(disp, variable=self._progress_var, maximum=100,
                             style="Heat.Horizontal.TProgressbar", length=260)
        pb.pack(pady=(0, 10), padx=12)

        # State label
        self._state_var = tk.StringVar(value=IDLE)
        tk.Label(frame, textvariable=self._state_var,
                 font=("Courier New", 12, "bold"), fg=C["accent"], bg=C["panel"]
                 ).pack(pady=(4, 2))

        # Confidence label (shown during SCANNING / CONFIRMED)
        self._conf_var = tk.StringVar(value="")
        tk.Label(frame, textvariable=self._conf_var,
                 font=("Courier New", 10), fg=C["green"], bg=C["panel"]
                 ).pack()

        # START button
        self._start_btn = tk.Button(frame, text="▶  START",
                                    font=("Courier New", 14, "bold"),
                                    fg=C["white"], bg=C["btn"],
                                    activebackground=C["btn_hover"],
                                    activeforeground=C["white"],
                                    relief="flat", bd=0, padx=24, pady=12,
                                    cursor="hand2",
                                    command=self._on_start)
        self._start_btn.pack(pady=(14, 6), ipadx=10)

        # BYPASS button (only visible during CONFIRMED)
        self._bypass_btn = tk.Button(frame, text="⚡  BYPASS HEATING",
                                     font=("Courier New", 11, "bold"),
                                     fg=C["white"], bg=C["amber"],
                                     activebackground="#a06010",
                                     activeforeground=C["white"],
                                     relief="flat", bd=0, padx=18, pady=9,
                                     cursor="hand2",
                                     command=self._on_bypass)

        # STOP button (only visible during HEATING)
        self._stop_btn = tk.Button(frame, text="⏹  STOP",
                                   font=("Courier New", 12, "bold"),
                                   fg=C["white"], bg=C["red"],
                                   activebackground="#c0392b",
                                   activeforeground=C["white"],
                                   relief="flat", bd=0, padx=20, pady=10,
                                   cursor="hand2",
                                   command=self._on_stop)

        # RESET button
        tk.Button(frame, text="↺  RESET",
                  font=("Courier New", 10), fg=C["dim"], bg=C["panel"],
                  activebackground=C["btn"], activeforeground=C["white"],
                  relief="flat", bd=0, padx=12, pady=6,
                  cursor="hand2",
                  command=self._on_reset).pack(pady=(0, 12))

        # Force simulation checkbox
        self._force_sim_var = tk.BooleanVar(value=False)
        tk.Checkbutton(frame, text="Force Simulation Mode",
                       variable=self._force_sim_var,
                       font=("Courier New", 9), fg=C["dim"], bg=C["panel"],
                       activebackground=C["panel"], activeforeground=C["white"],
                       selectcolor=C["btn"], relief="flat",
                       command=self._on_toggle_sim).pack(pady=(0, 4))

        # Simulation mode label
        sim_label = ("SIMULATION MODE  —  no webcam"
                     if self.detector.sim_mode else
                     "LIVE MODE  —  webcam + YOLO active")
        tk.Label(frame, text=sim_label, font=("Courier New", 8),
                 fg=C["dim"], bg=C["panel"]).pack(side="bottom", pady=6)

    # ══════════════════════════════════════════════════════════════════════════
    # State Machine
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_idle(self):
        self.state = IDLE
        self._food = None
        self._state_var.set("[ IDLE ]")
        self._food_lbl.config(text="---", fg=C["green"])
        self._food_desc.config(text="Insert food & press START")
        self._temp_var.set("-- °F")
        self._progress_var.set(0)
        self._conf_var.set("")
        self._set_status("Ready. Place food inside and press START.")
        self._start_btn.config(text="▶  START", state="normal", fg=C["white"])

    def _enter_scanning(self):
        self.state = SCANNING
        self._bypass_btn.pack_forget()
        self._state_var.set("[ SCANNING ]")
        self._food_lbl.config(text="🔍", fg=C["amber"])
        self._food_desc.config(text="Scanning food — please wait…")
        self._conf_var.set("")
        self._set_status("Camera is scanning the food inside the microwave…")
        self._start_btn.config(state="disabled")
        threading.Thread(target=self._run_scan, daemon=True).start()

    def _enter_confirmed(self, food: dict):
        self.state = CONFIRMED
        self._food = food
        self.thermal.reset(initial_temp_f=food["initial_temp_f"],
                           heat_seconds=self._heat_seconds)
        self._state_var.set("[ FOOD IDENTIFIED ]")
        self._food_lbl.config(text=f"{food['icon']} {food['name'].upper()}", fg=C["green"])
        self._food_desc.config(text=food["desc"])
        self._conf_var.set("✔  Identification confirmed")
        self._set_status(
            f"Detected: {food['desc']}. "
            f"Press START again to begin heating to 165 °F."
        )
        self._start_btn.config(text="▶  START HEATING", state="normal", fg=C["green"])
        self._bypass_btn.pack(pady=(0, 6), ipadx=8)

    def _enter_heating(self):
        self.state = HEATING
        self._bypass_btn.pack_forget()
        self._state_var.set("[ HEATING ]")
        self._start_btn.config(state="disabled", text="HEATING…")
        self._conf_var.set("")
        self._stop_btn.pack(pady=(0, 6), ipadx=8)
        self.thermal.start()
        if self.thermal.force_sim:
            self._set_status("Simulating heating to 165 °F — no microwave required.")
        else:
            self._set_status("Heating food. Infrared camera monitoring temperature…")
        threading.Thread(target=self._run_heating, daemon=True).start()

    def _enter_complete(self):
        self.state = COMPLETE
        self._state_var.set("[ COMPLETE ]")
        self._progress_var.set(100)
        self._stop_btn.pack_forget()
        self._food_lbl.config(fg=C["red"])
        self._set_status("✅ Food has reached 165 °F — safe to eat! Press RESET to start over.")
        self._start_btn.config(text="↺  RESET", state="normal",
                               fg=C["amber"], command=self._on_reset)
        self._play_done_sound()
        # Return to live camera feed 10 s after completion
        self._sim_clear_timer = self.after(10000, self.detector.clear_sim_frame)

    # ══════════════════════════════════════════════════════════════════════════
    # Background Workers
    # ══════════════════════════════════════════════════════════════════════════

    def _run_scan(self):
        food = self.detector.scan(duration=3.0)
        if food:
            self.after(0, lambda: self._enter_confirmed(food))
        else:
            self.after(0, self._enter_idle)
            self.after(0, lambda: self._set_status(
                "No food detected. Please place food inside and try again."))

    def _run_heating(self):
        while self._running and self.state == HEATING:
            temp  = self.thermal.temperature
            pct   = max(0.0, min(100.0,
                        (temp - self._food["initial_temp_f"]) /
                        (ThermalCamera.TARGET_F - self._food["initial_temp_f"]) * 100))
            self.after(0, lambda t=temp, p=pct: self._update_heat_display(t, p))
            if self.thermal.reached_target:
                self.after(0, self._enter_complete)
                return
            time.sleep(0.25)

    def _update_heat_display(self, temp: float, pct: float):
        self._temp_var.set(f"{temp:.1f} °F")
        self._progress_var.set(pct)

    # ══════════════════════════════════════════════════════════════════════════
    # Refresh Loop (runs on main thread via after())
    # ══════════════════════════════════════════════════════════════════════════

    def _tick(self):
        if not self._running:
            return
        self._refresh_camera()
        if self.state in (HEATING, COMPLETE):
            self._refresh_ir()
        else:
            self._refresh_ir_standby()
        self.after(66, self._tick)   # ~15 fps

    def _refresh_camera(self):
        frame = self.detector.grab_frame()
        if frame is None:
            return
        frame = cv2.resize(frame, (400, 300))
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self._cam_canvas.create_image(0, 0, anchor="nw", image=img)
        self._cam_canvas._img = img   # keep reference

    def _refresh_ir(self):
        ir = self.thermal.colormap_frame(width=400, height=120)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)))
        self._ir_canvas.create_image(0, 0, anchor="nw", image=img)
        self._ir_canvas._img = img

    def _refresh_ir_standby(self):
        self._ir_canvas.delete("all")
        self._ir_canvas.create_rectangle(0, 0, 400, 120, fill="#050a10", outline="")
        self._ir_canvas.create_text(200, 60, text="THERMAL  —  STANDBY",
                                    font=("Courier New", 11), fill=C["dim"])

    # ══════════════════════════════════════════════════════════════════════════
    # Button Handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _on_start(self):
        if self.state == IDLE:
            self._enter_scanning()
        elif self.state == CONFIRMED:
            self._enter_heating()

    def _on_bypass(self):
        self.thermal.force_sim = True
        self._enter_heating()

    def _on_reset(self):
        if self._sim_clear_timer:
            self.after_cancel(self._sim_clear_timer)
            self._sim_clear_timer = None
        self._bypass_btn.pack_forget()
        self._stop_btn.pack_forget()
        self._start_btn.config(command=self._on_start)
        self.thermal.force_sim = False
        self.detector.clear_sim_frame()
        self._enter_idle()

    def _on_stop(self):
        self.thermal.stop()                        # freeze temperature immediately
        self.state = COMPLETE                      # exit heating loop
        self._stop_btn.pack_forget()
        stopped_temp = self.thermal.temperature
        self._state_var.set("[ STOPPED ]")
        self._temp_var.set(f"{stopped_temp:.1f} °F")
        self._set_status(
            f"Heating stopped at {stopped_temp:.1f} °F. "
            f"Continue reheating or press RESET to start over."
        )
        self._start_btn.config(text="▶  CONTINUE HEATING", state="normal",
                               fg=C["green"], command=self._on_continue)

    def _on_continue(self):
        self.thermal.resume(heat_seconds_remaining=20.0)
        self._enter_heating()

    def _on_toggle_sim(self):
        self.detector.force_sim = self._force_sim_var.get()

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _set_status(self, msg: str):
        self._status_var.set(msg)

    @staticmethod
    def _play_done_sound():
        """Play a completion chime (cross-platform)."""
        system = platform.system()
        try:
            if system == "Windows":
                import winsound
                for freq, dur in [(880, 200), (1046, 200), (1318, 400)]:
                    winsound.Beep(freq, dur)
            elif system == "Darwin":
                import subprocess
                subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
            else:
                import subprocess
                subprocess.Popen(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"])
        except Exception:
            pass   # sound is best-effort

    def _on_close(self):
        self._running = False
        self.detector.release()
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SmartMicrowaveApp()
    app.mainloop()
