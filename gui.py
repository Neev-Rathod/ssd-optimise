# -*- coding: utf-8 -*-
"""
gui.py
======
Tkinter GUI front-end for the AI-Era SSD Emulator benchmark.

Layout
------
  +--------------------------------------------------+
  | HEADER  (title + Run button + status badge)      |
  +--------------------------------------------------+
  | PHASE PANEL  (4 phase cards, animated progress)  |
  +--------------------------------------------------+
  | LIVE LOG  (scrolling activity feed)              |
  +--------------------------------------------------+
  |  CHART AREA        |  RESULTS TABLE              |
  +--------------------+-----------------------------+

The benchmark runs on a background thread.  Progress events are pushed
through a queue and consumed by an after()-loop on the main thread,
keeping the GUI fully responsive.

Run:
    python gui.py
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import emulator
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Redirect emulator's logging through our queue BEFORE importing emulator
import logging

class _QueueHandler(logging.Handler):
    """Push log records into a queue for consumption by the GUI thread."""
    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        self._q.put(("log", self.format(record)))


# We patch matplotlib backend before emulator imports it
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Now import emulator (it will use our patched matplotlib backend)
import emulator as emu

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG          = "#0f1117"   # near-black background
BG2         = "#1a1d2e"   # slightly lighter card bg
BG3         = "#232740"   # panel / log background
ACCENT_BLUE = "#4CA6E8"   # AI-SSD / optimised
ACCENT_RED  = "#E55C5C"   # Baseline
ACCENT_GRN  = "#4CD97B"   # success green
ACCENT_YLW  = "#F5C542"   # warning / running yellow
FG          = "#E8EAF0"   # primary foreground
FG2         = "#8890A8"   # secondary foreground
BORDER      = "#2e3350"   # card border colour

FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_HEAD   = ("Segoe UI", 11, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_MONO   = ("Consolas", 9)
FONT_BADGE  = ("Segoe UI", 9, "bold")
FONT_METRIC = ("Segoe UI", 13, "bold")
FONT_LABEL  = ("Segoe UI", 9)

# ---------------------------------------------------------------------------
# Phase definitions  (id, label, description)
# ---------------------------------------------------------------------------
PHASES = [
    ("gen",   "Phase 0",  "Data Generator",
     "Generating 100 MB binary model-checkpoint file with float32 tensor blocks."),
    ("load_b","Phase 1A", "Baseline Dataset Read",
     "Standard OS open()+read() in 4 KB chunks — full user-space copy path."),
    ("load_o","Phase 1B", "AI-SSD Dataset Read",
     "mmap + readahead warm-touch — zero-copy numpy views from page cache."),
    ("kv_b",  "Phase 2A", "Baseline KV-Cache Inference",
     "Synchronous evict-and-reload on every step — no prefetch, 0% hit rate."),
    ("kv_o",  "Phase 2B", "Optimised KV-Cache Inference",
     "Async predictive prefetch worker — next block pre-loaded behind compute."),
]

# ===========================================================================
# Helpers
# ===========================================================================

def _make_frame(parent: tk.Widget, bg: str = BG2, bd: int = 1,
                relief: str = "flat", **kw: Any) -> tk.Frame:
    f = tk.Frame(parent, bg=bg, bd=bd, relief=relief,
                 highlightbackground=BORDER, highlightthickness=1, **kw)
    return f


def _label(parent: tk.Widget, text: str, fg: str = FG, bg: str = BG2,
           font: Any = FONT_BODY, **kw: Any) -> tk.Label:
    return tk.Label(parent, text=text, fg=fg, bg=bg, font=font, **kw)


# ===========================================================================
# Main Application Window
# ===========================================================================

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AI-Era SSD Emulator")
        self.configure(bg=BG)
        self.minsize(1100, 760)
        self.geometry("1200x820")

        # Try to maximise on start
        try:
            self.state("zoomed")
        except Exception:
            pass

        # Queue used by worker thread to send events to the GUI
        self._q: queue.Queue = queue.Queue()

        # Install queue log handler into emulator's logger
        self._log_handler = _QueueHandler(self._q)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        )
        logging.getLogger("emulator").addHandler(self._log_handler)
        logging.getLogger("emulator").setLevel(logging.DEBUG)

        # State
        self._running     = False
        self._baseline:   Optional[emu.BenchmarkResult] = None
        self._optimised:  Optional[emu.BenchmarkResult] = None
        self._phase_vars: Dict[str, Dict[str, Any]] = {}   # per-phase state
        self._hardware_vars: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._guide_index = -1
        self._guide_var: Optional[tk.StringVar] = None

        # Build UI
        self._build_ui()

        # Start queue polling
        self._poll_queue()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True)

        # Header
        self._build_header(outer)

        # Scrollable content area so the complete dashboard fits small screens.
        scroll_shell = tk.Frame(outer, bg=BG)
        scroll_shell.pack(fill="both", expand=True)
        canvas = tk.Canvas(scroll_shell, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_shell, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        content = tk.Frame(canvas, bg=BG)
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>",
                 lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                lambda event: canvas.itemconfigure(content_window, width=event.width))
        canvas.bind_all("<MouseWheel>",
                lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"))
        content.pack_propagate(False)
        content.configure(width=1200)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=0)   # phase panel
        content.rowconfigure(1, weight=0)   # hardware path
        content.rowconfigure(2, weight=1)   # log + chart/table

        # Phase panel (full width, row 0)
        self._build_phase_panel(content)

        # Hardware path (full width, row 1)
        self._build_hardware_panel(content)

        # Left column: live log
        self._build_log_panel(content)

        # Right column: chart + results table (stacked)
        self._build_right_panel(content)

    def _build_header(self, parent: tk.Widget) -> None:
        hdr = tk.Frame(parent, bg=BG3, pady=14)
        hdr.pack(fill="x", padx=0, pady=(0, 2))

        # Title
        tk.Label(hdr, text="AI-Era SSD Emulator", font=FONT_TITLE,
                 fg=ACCENT_BLUE, bg=BG3).pack(side="left", padx=20)
        tk.Label(hdr, text="Benchmark & Visualiser", font=("Segoe UI", 13),
                 fg=FG2, bg=BG3).pack(side="left", padx=(0, 30))

        # Status badge (right side)
        self._status_var = tk.StringVar(value="Ready")
        self._status_lbl = tk.Label(hdr, textvariable=self._status_var,
                                    font=FONT_BADGE, fg=ACCENT_GRN, bg=BG3,
                                    padx=12, pady=4)
        self._status_lbl.pack(side="right", padx=20)

        # Run button
        self._run_btn = tk.Button(
            hdr, text="  Run Benchmark  ",
            font=("Segoe UI", 11, "bold"),
            fg="white", bg=ACCENT_BLUE,
            activebackground="#3a8fd4", activeforeground="white",
            relief="flat", bd=0, padx=16, pady=6, cursor="hand2",
            command=self._start_benchmark,
        )
        self._run_btn.pack(side="right", padx=6)

        # Reset button
        tk.Button(
            hdr, text="Reset",
            font=("Segoe UI", 10),
            fg=FG2, bg=BG3,
            activebackground=BG2, activeforeground=FG,
            relief="flat", bd=0, padx=10, pady=6, cursor="hand2",
            command=self._reset_ui,
        ).pack(side="right", padx=2)

        self._next_btn = tk.Button(
            hdr, text="Next Step",
            font=("Segoe UI", 10, "bold"),
            fg=FG, bg=BG2, activebackground=ACCENT_BLUE,
            activeforeground="white", relief="flat", bd=0,
            padx=10, pady=6, cursor="hand2", command=self._next_guide_step,
        )
        self._next_btn.pack(side="right", padx=2)

    def _build_phase_panel(self, parent: tk.Widget) -> None:
        frame = _make_frame(parent, bg=BG2)
        frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
        frame.columnconfigure(list(range(len(PHASES))), weight=1)

        tk.Label(frame, text="Benchmark Phases", font=FONT_HEAD,
                 fg=FG, bg=BG2).grid(row=0, column=0, columnspan=len(PHASES),
                                     sticky="w", padx=12, pady=(8, 4))

        for col, (pid, phase_lbl, title, desc) in enumerate(PHASES):
            card = _make_frame(frame, bg=BG3)
            card.grid(row=1, column=col, sticky="nsew", padx=6, pady=(0, 10))
            card.columnconfigure(0, weight=1)

            # Phase label pill
            pill = tk.Frame(card, bg=ACCENT_BLUE, pady=2)
            pill.grid(row=0, column=0, sticky="ew")
            tk.Label(pill, text=phase_lbl, font=FONT_BADGE,
                     fg="white", bg=ACCENT_BLUE).pack()

            # Title
            tk.Label(card, text=title, font=("Segoe UI", 9, "bold"),
                     fg=FG, bg=BG3, wraplength=180, justify="center"
                     ).grid(row=1, column=0, padx=8, pady=(6, 2))

            # Description
            tk.Label(card, text=desc, font=("Segoe UI", 8),
                     fg=FG2, bg=BG3, wraplength=180, justify="center"
                     ).grid(row=2, column=0, padx=8, pady=(0, 6))

            # Progress bar
            pb = ttk.Progressbar(card, mode="indeterminate", length=160)
            pb.grid(row=3, column=0, padx=10, pady=(0, 4))
            pb_style = ttk.Style()
            pb_style.theme_use("default")
            pb_style.configure("TProgressbar", background=ACCENT_BLUE,
                                troughcolor=BG2, thickness=5)

            # Status label below bar
            sv = tk.StringVar(value="Waiting")
            sl = tk.Label(card, textvariable=sv, font=FONT_LABEL,
                          fg=FG2, bg=BG3)
            sl.grid(row=4, column=0, pady=(0, 8))

            # Metric display (shown after phase completes)
            mv = tk.StringVar(value="")
            ml = tk.Label(card, textvariable=mv, font=FONT_METRIC,
                          fg=ACCENT_GRN, bg=BG3)
            ml.grid(row=5, column=0, pady=(0, 10))

            self._phase_vars[pid] = {
                "pill": pill, "pb": pb, "sv": sv, "sl": sl, "mv": mv, "card": card
            }

    def _build_hardware_panel(self, parent: tk.Widget) -> None:
        frame = _make_frame(parent, bg=BG2)
        frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 6))
        frame.columnconfigure(0, weight=1)

        tk.Label(frame, text="Live Hardware Paths", font=FONT_HEAD,
                 fg=FG, bg=BG2).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 2))
        self._guide_var = tk.StringVar(
            value="Press Next Step to inspect the data flow, or run the benchmark for live activity.")
        tk.Label(frame, textvariable=self._guide_var, font=FONT_LABEL,
             fg=FG2, bg=BG2).grid(row=0, column=0, sticky="e", padx=12, pady=(8, 2))

        lanes = {
            "normal": ("NORMAL / STANDARD I/O", ACCENT_RED,
                       [("app", "MODEL / APP", "Request"), ("cpu", "CPU", "Compute"),
                        ("ram", "RAM", "KV-cache"), ("page", "OS PAGE CACHE", "Copy buffer"),
                        ("ssd", "SSD STORAGE", "4 KB reads")]),
            "ai": ("AI-SSD OPTIMISED I/O", ACCENT_BLUE,
                   [("app", "MODEL / APP", "Request"), ("cpu", "CPU", "Compute"),
                    ("ram", "RAM", "KV-cache"), ("prefetch", "PREFETCH ENGINE", "Next block"),
                    ("page", "OS PAGE CACHE", "Warm pages"), ("ssd", "SSD STORAGE", "mmap reads")]),
        }
        for row, (lane_key, (lane_title, lane_colour, hardware)) in enumerate(lanes.items(), start=1):
            tk.Label(frame, text=lane_title, font=("Segoe UI", 9, "bold"),
                     fg=lane_colour, bg=BG2).grid(row=row, column=0, sticky="w", padx=12, pady=(3, 0))
            path = tk.Frame(frame, bg=BG2)
            path.grid(row=row, column=0, sticky="ew", padx=8, pady=(0, 5))
            self._hardware_vars[lane_key] = {}
            for index, (key, title, detail) in enumerate(hardware):
                path.columnconfigure(index * 2, weight=1)
                card = tk.Frame(path, bg=BG3, highlightbackground=BORDER,
                                highlightthickness=1, padx=8, pady=4)
                card.grid(row=0, column=index * 2, sticky="ew")
                title_label = tk.Label(card, text=title, font=("Segoe UI", 8, "bold"),
                                       fg=FG, bg=BG3)
                title_label.pack()
                detail_label = tk.Label(card, text=detail, font=("Segoe UI", 8),
                                        fg=FG2, bg=BG3)
                detail_label.pack()
                status = tk.Label(card, text="Waiting", font=FONT_LABEL,
                                  fg=FG2, bg=BG3)
                status.pack(pady=(1, 0))
                pulse = tk.Frame(card, height=3, bg=BORDER)
                pulse.pack(fill="x", pady=(2, 0))
                self._hardware_vars[lane_key][key] = {
                    "card": card, "status": status, "pulse": pulse,
                    "colour": lane_colour, "labels": (title_label, detail_label),
                }
                if index < len(hardware) - 1:
                    tk.Label(path, text=">", font=("Segoe UI", 14, "bold"),
                             fg=FG2, bg=BG2).grid(row=0, column=index * 2 + 1, padx=4)

    def _build_log_panel(self, parent: tk.Widget) -> None:
        frame = _make_frame(parent, bg=BG2)
        frame.grid(row=2, column=0, sticky="nsew", padx=(4, 2), pady=4)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        tk.Label(frame, text="Live Activity Feed", font=FONT_HEAD,
                 fg=FG, bg=BG2).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 2))

        # Text widget + scrollbar
        txt_frame = tk.Frame(frame, bg=BG3)
        txt_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        txt_frame.rowconfigure(0, weight=1)
        txt_frame.columnconfigure(0, weight=1)

        self._log_txt = tk.Text(
            txt_frame,
            bg=BG3, fg=FG, font=FONT_MONO,
            insertbackground=FG,
            selectbackground=ACCENT_BLUE,
            relief="flat", bd=0,
            state="disabled", wrap="word",
            spacing1=2, spacing3=2,
        )
        self._log_txt.grid(row=0, column=0, sticky="nsew")

        sb = tk.Scrollbar(txt_frame, command=self._log_txt.yview, bg=BG3,
                          troughcolor=BG3, activebackground=ACCENT_BLUE)
        sb.grid(row=0, column=1, sticky="ns")
        self._log_txt["yscrollcommand"] = sb.set

        # Text colour tags
        self._log_txt.tag_configure("info",    foreground=FG)
        self._log_txt.tag_configure("phase",   foreground=ACCENT_BLUE,  font=("Consolas", 9, "bold"))
        self._log_txt.tag_configure("metric",  foreground=ACCENT_GRN,   font=("Consolas", 9, "bold"))
        self._log_txt.tag_configure("warn",    foreground=ACCENT_YLW)
        self._log_txt.tag_configure("err",     foreground=ACCENT_RED,   font=("Consolas", 9, "bold"))
        self._log_txt.tag_configure("sep",     foreground=FG2)
        self._log_txt.tag_configure("ts",      foreground=FG2,          font=("Consolas", 8))

    def _build_right_panel(self, parent: tk.Widget) -> None:
        right = tk.Frame(parent, bg=BG)
        right.grid(row=2, column=1, sticky="nsew", padx=(2, 4), pady=4)
        right.rowconfigure(0, weight=3)   # chart gets more space
        right.rowconfigure(1, weight=2)   # results table
        right.columnconfigure(0, weight=1)

        self._build_chart_panel(right)
        self._build_results_panel(right)

    def _build_chart_panel(self, parent: tk.Widget) -> None:
        frame = _make_frame(parent, bg=BG2)
        frame.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        tk.Label(frame, text="Live Charts", font=FONT_HEAD,
                 fg=FG, bg=BG2).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 2))

        # Placeholder canvas — will be replaced by embedded matplotlib figure
        self._chart_placeholder = tk.Frame(frame, bg=BG3, height=300)
        self._chart_placeholder.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self._chart_lbl = tk.Label(
            self._chart_placeholder,
            text="Charts will appear here after the benchmark completes.",
            font=FONT_BODY, fg=FG2, bg=BG3,
        )
        self._chart_lbl.place(relx=0.5, rely=0.5, anchor="center")

        self._chart_frame = frame  # keep ref for embedding canvas

    def _build_results_panel(self, parent: tk.Widget) -> None:
        frame = _make_frame(parent, bg=BG2)
        frame.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        tk.Label(frame, text="Results Summary", font=FONT_HEAD,
                 fg=FG, bg=BG2).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 2))

        tbl_frame = tk.Frame(frame, bg=BG3)
        tbl_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        tbl_frame.rowconfigure(0, weight=1)
        tbl_frame.columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.Treeview",
                        background=BG3, foreground=FG,
                        fieldbackground=BG3,
                        rowheight=24, font=("Segoe UI", 9),
                        bordercolor=BORDER, borderwidth=0)
        style.configure("Dark.Treeview.Heading",
                        background=BG2, foreground=FG2,
                        font=("Segoe UI", 9, "bold"),
                        relief="flat")
        style.map("Dark.Treeview",
                  background=[("selected", ACCENT_BLUE)],
                  foreground=[("selected", "white")])

        cols = ("metric", "baseline", "optimised", "speedup")
        self._tree = ttk.Treeview(
            tbl_frame, columns=cols, show="headings",
            style="Dark.Treeview", height=9,
        )
        self._tree.heading("metric",    text="Metric")
        self._tree.heading("baseline",  text="Baseline")
        self._tree.heading("optimised", text="AI-SSD Optimised")
        self._tree.heading("speedup",   text="Improvement")

        self._tree.column("metric",    width=200, anchor="w")
        self._tree.column("baseline",  width=130, anchor="center")
        self._tree.column("optimised", width=150, anchor="center")
        self._tree.column("speedup",   width=120, anchor="center")

        self._tree.tag_configure("good",    background="#1a2e1a", foreground=ACCENT_GRN)
        self._tree.tag_configure("neutral", background=BG3,       foreground=FG)
        self._tree.tag_configure("section", background=BG2,       foreground=FG2,
                                 font=("Segoe UI", 8, "bold"))

        vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=self._tree.yview)
        self._tree["yscrollcommand"] = vsb.set

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

    # -----------------------------------------------------------------------
    # Benchmark control
    # -----------------------------------------------------------------------

    def _start_benchmark(self) -> None:
        if self._running:
            return
        self._reset_ui(keep_log=True)
        self._running = True
        self._run_btn.config(state="disabled", bg="#2a5a7a")
        self._set_status("Running...", ACCENT_YLW)
        self._log_append("=" * 60, "sep")
        self._log_append("  Benchmark started", "phase")
        self._log_append("=" * 60, "sep")

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self) -> None:
        """Background thread: run the full benchmark, push events to queue."""
        try:
            # ---- Phase 0: data generation --------------------------------
            self._q.put(("phase_start", "gen"))
            dataset_path = emu.generate_dataset()
            sz = dataset_path.stat().st_size / (1024 ** 2)
            self._q.put(("phase_done", "gen", f"{sz:.0f} MB ready"))

            # ---- Phase 1A: baseline load ---------------------------------
            self._q.put(("phase_start", "load_b"))
            import gc
            gc.collect()
            std_reader = emu.StandardReader(dataset_path)
            elapsed_b, tp_b, ram_b = std_reader.read_all()
            self._q.put(("phase_done", "load_b",
                         f"{elapsed_b*1000:.0f} ms  |  {tp_b:.0f} MB/s"))

            # ---- Phase 1B: AI-SSD load -----------------------------------
            self._q.put(("phase_start", "load_o"))
            gc.collect()
            ai_reader = emu.AISSDReader(dataset_path)
            elapsed_o, tp_o, ram_o = ai_reader.read_all()
            self._q.put(("phase_done", "load_o",
                         f"{elapsed_o*1000:.0f} ms  |  {tp_o:.0f} MB/s"))

            # ---- Phase 2A: baseline KV-cache -----------------------------
            self._q.put(("phase_start", "kv_b"))
            std_reader.clear_cache()
            gc.collect()
            baseline_kv = emu.BaselineKVCache(std_reader)
            kv_lats_b, kv_ram_b, kv_hit_b = baseline_kv.run_inference()
            avg_b = float(np.mean(kv_lats_b))
            self._q.put(("phase_done", "kv_b",
                         f"{avg_b:.1f} ms/step  |  {kv_hit_b:.0f}% hit"))

            # ---- Phase 2B: optimised KV-cache ----------------------------
            self._q.put(("phase_start", "kv_o"))
            gc.collect()
            with emu.AISSDReader(dataset_path) as opt_reader:
                opt_kv = emu.OptimisedKVCache(opt_reader)
                kv_lats_o, kv_ram_o, kv_hit_o = opt_kv.run_inference()
            avg_o = float(np.mean(kv_lats_o))
            self._q.put(("phase_done", "kv_o",
                         f"{avg_o:.1f} ms/step  |  {kv_hit_o:.0f}% hit"))

            # ---- Package results -----------------------------------------
            baseline = emu.BenchmarkResult(
                label="Baseline (Standard I/O)",
                dataset_latency_ms=elapsed_b * 1000,
                throughput_mb_s=tp_b,
                peak_ram_mb=ram_b,
                kv_step_latencies=kv_lats_b,
                kv_peak_ram_mb=kv_ram_b,
                kv_hit_rate_pct=kv_hit_b,
            )
            optimised = emu.BenchmarkResult(
                label="Optimised (AI-SSD Emulated)",
                dataset_latency_ms=elapsed_o * 1000,
                throughput_mb_s=tp_o,
                peak_ram_mb=ram_o,
                kv_step_latencies=kv_lats_o,
                kv_peak_ram_mb=kv_ram_o,
                kv_hit_rate_pct=kv_hit_o,
            )

            self._q.put(("results", baseline, optimised))

        except Exception as exc:
            import traceback
            self._q.put(("error", traceback.format_exc()))

    # -----------------------------------------------------------------------
    # Queue polling  (runs on main/GUI thread)
    # -----------------------------------------------------------------------

    def _poll_queue(self) -> None:
        try:
            while True:
                event = self._q.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass
        self.after(60, self._poll_queue)

    def _handle_event(self, event: tuple) -> None:
        kind = event[0]

        if kind == "log":
            msg = event[1]
            tag = "info"
            if "Phase" in msg or "BENCHMARK" in msg or "===" in msg:
                tag = "phase"
            elif "Latency" in msg or "Throughput" in msg or "Hit" in msg or "Avg" in msg:
                tag = "metric"
            elif "WARNING" in msg or "WARN" in msg:
                tag = "warn"
            elif "ERROR" in msg:
                tag = "err"
            elif "---" in msg or "===" in msg:
                tag = "sep"
            self._log_append(msg, tag)

        elif kind == "phase_start":
            pid = event[1]
            self._phase_set_running(pid)
            self._hardware_set_phase(pid)

        elif kind == "phase_done":
            pid, metric = event[1], event[2]
            self._phase_set_done(pid, metric)
            self._hardware_set_done(pid)

        elif kind == "results":
            baseline, optimised = event[1], event[2]
            self._baseline  = baseline
            self._optimised = optimised
            self._running   = False
            self._run_btn.config(state="normal", bg=ACCENT_BLUE)
            self._set_status("Complete!", ACCENT_GRN)
            self._log_append("=" * 60, "sep")
            self._log_append("  Benchmark complete!", "phase")
            self._log_append("=" * 60, "sep")
            self._populate_results(baseline, optimised)
            self._embed_chart(baseline, optimised)

        elif kind == "error":
            tb = event[1]
            self._running = False
            self._run_btn.config(state="normal", bg=ACCENT_BLUE)
            self._set_status("Error!", ACCENT_RED)
            self._log_append("ERROR:\n" + tb, "err")

    # -----------------------------------------------------------------------
    # Phase card state management
    # -----------------------------------------------------------------------

    def _phase_set_running(self, pid: str) -> None:
        v = self._phase_vars.get(pid)
        if not v:
            return
        v["pill"].config(bg=ACCENT_YLW)
        for w in v["pill"].winfo_children():
            w.config(bg=ACCENT_YLW, fg=BG)
        v["sv"].set("Running...")
        v["sl"].config(fg=ACCENT_YLW)
        v["pb"].start(12)

    def _phase_set_done(self, pid: str, metric: str) -> None:
        v = self._phase_vars.get(pid)
        if not v:
            return
        v["pill"].config(bg=ACCENT_GRN)
        for w in v["pill"].winfo_children():
            w.config(bg=ACCENT_GRN, fg=BG)
        v["pb"].stop()
        v["sv"].set("Done")
        v["sl"].config(fg=ACCENT_GRN)
        v["mv"].set(metric)

    def _phase_reset(self, pid: str) -> None:
        v = self._phase_vars.get(pid)
        if not v:
            return
        v["pill"].config(bg=ACCENT_BLUE)
        for w in v["pill"].winfo_children():
            w.config(bg=ACCENT_BLUE, fg="white")
        v["pb"].stop()
        v["sv"].set("Waiting")
        v["sl"].config(fg=FG2)
        v["mv"].set("")

    def _hardware_set_phase(self, pid: str) -> None:
        active = {
            "gen": {"normal": {"app", "ssd"}, "ai": {"app", "ssd"}},
            "load_b": {"normal": {"app", "cpu", "ram", "page", "ssd"}, "ai": set()},
            "load_o": {"normal": set(), "ai": {"app", "ram", "page", "ssd", "prefetch"}},
            "kv_b": {"normal": {"app", "cpu", "ram", "page", "ssd"}, "ai": set()},
            "kv_o": {"normal": set(), "ai": {"app", "cpu", "ram", "page", "ssd", "prefetch"}},
        }.get(pid, {})
        self._set_hardware_activity(active, "ACTIVE")
        explanations = {
            "gen": "Both lanes use the SSD to create or verify the model checkpoint.",
            "load_b": "Normal I/O copies many small chunks from SSD through the OS into RAM.",
            "load_o": "AI-SSD maps the file and prefetches the next pages before they are requested.",
            "kv_b": "Normal inference waits for each KV block before CPU computation continues.",
            "kv_o": "AI-SSD inference computes while the prefetch engine prepares the next KV block.",
        }
        if self._guide_var is not None:
            self._guide_var.set(explanations.get(pid, "Benchmark activity is visible in both hardware lanes."))

    def _hardware_set_done(self, pid: str) -> None:
        active = {
            "gen": {"normal": {"ssd"}, "ai": {"ssd"}},
            "load_b": {"normal": {"ram", "page"}, "ai": set()},
            "load_o": {"normal": set(), "ai": {"ram", "page"}},
            "kv_b": {"normal": {"ram"}, "ai": set()},
            "kv_o": {"normal": set(), "ai": {"ram", "prefetch"}},
        }.get(pid, {})
        self._set_hardware_activity(active, "READY")

    def _set_hardware_activity(self, active: Dict[str, set], active_text: str) -> None:
        for lane, blocks in self._hardware_vars.items():
            for key, values in blocks.items():
                is_active = key in active.get(lane, set())
                colour = values["colour"]
                values["card"].config(bg=colour if is_active else BG3)
                values["status"].config(text=active_text if is_active else "Standby",
                                         fg=BG if is_active else FG2,
                                         bg=colour if is_active else BG3)
                values["pulse"].config(bg=colour if is_active else BORDER)
                for index, label in enumerate(values["labels"]):
                    label.config(bg=colour if is_active else BG3,
                                 fg=BG if is_active else (FG if index == 0 else FG2))

    def _hardware_reset(self) -> None:
        self._set_hardware_activity({}, "Waiting")
        if self._guide_var is not None:
            self._guide_var.set("Press Next Step to inspect the data flow, or run the benchmark for live activity.")

    def _next_guide_step(self) -> None:
        steps = [
            ("normal", {"app", "ssd"}, "Step 1/5: the application requests model data from storage."),
            ("normal", {"page", "ram"}, "Step 2/5: normal I/O copies data through the OS page cache into RAM."),
            ("normal", {"cpu", "ram"}, "Step 3/5: the CPU consumes the RAM-resident KV-cache block."),
            ("ai", {"ssd", "prefetch", "page"}, "Step 4/5: AI-SSD maps storage and prefetches the next block."),
            ("ai", {"cpu", "ram", "prefetch"}, "Step 5/5: CPU compute overlaps with the next-block prefetch."),
        ]
        self._guide_index = (self._guide_index + 1) % len(steps)
        lane, active_blocks, explanation = steps[self._guide_index]
        self._set_hardware_activity({lane: active_blocks}, "ACTIVE")
        if self._guide_var is not None:
            self._guide_var.set(explanation)

    # -----------------------------------------------------------------------
    # Log helpers
    # -----------------------------------------------------------------------

    def _log_append(self, text: str, tag: str = "info") -> None:
        self._log_txt.config(state="normal")
        self._log_txt.insert("end", text + "\n", tag)
        self._log_txt.see("end")
        self._log_txt.config(state="disabled")

    # -----------------------------------------------------------------------
    # Status badge
    # -----------------------------------------------------------------------

    def _set_status(self, text: str, colour: str = ACCENT_GRN) -> None:
        self._status_var.set(text)
        self._status_lbl.config(fg=colour)

    # -----------------------------------------------------------------------
    # Results table
    # -----------------------------------------------------------------------

    def _populate_results(self, b: emu.BenchmarkResult, o: emu.BenchmarkResult) -> None:
        # Clear existing rows
        for row in self._tree.get_children():
            self._tree.delete(row)

        def _su(bv: float, ov: float, higher: bool = False) -> str:
            if ov == 0:
                return "-"
            ratio = (ov / max(bv, 1e-9)) if higher else (bv / max(ov, 1e-9))
            arrow = "+" if higher else "-"
            return f"{ratio:.1f}x {arrow}"

        def _tag(bv: float, ov: float, higher: bool = False) -> str:
            ratio = (ov / max(bv, 1e-9)) if higher else (bv / max(ov, 1e-9))
            return "good" if ratio > 1.2 else "neutral"

        rows = [
            ("--- Dataset Load ---", "", "", "", "section"),
            ("Load Latency",
             f"{b.dataset_latency_ms:.1f} ms",
             f"{o.dataset_latency_ms:.1f} ms",
             _su(b.dataset_latency_ms, o.dataset_latency_ms),
             _tag(b.dataset_latency_ms, o.dataset_latency_ms)),
            ("Read Throughput",
             f"{b.throughput_mb_s:.0f} MB/s",
             f"{o.throughput_mb_s:.0f} MB/s",
             _su(b.throughput_mb_s, o.throughput_mb_s, higher=True),
             _tag(b.throughput_mb_s, o.throughput_mb_s, higher=True)),
            ("Peak RAM Delta",
             f"{b.peak_ram_mb:.1f} MB",
             f"{o.peak_ram_mb:.1f} MB",
             "",
             "neutral"),
            ("--- KV-Cache Inference ---", "", "", "", "section"),
            ("Avg Step Latency",
             f"{b.avg_kv_latency_ms:.2f} ms/step",
             f"{o.avg_kv_latency_ms:.2f} ms/step",
             _su(b.avg_kv_latency_ms, o.avg_kv_latency_ms),
             _tag(b.avg_kv_latency_ms, o.avg_kv_latency_ms)),
            ("p99 Step Latency",
             f"{b.p99_kv_latency_ms:.2f} ms",
             f"{o.p99_kv_latency_ms:.2f} ms",
             _su(b.p99_kv_latency_ms, o.p99_kv_latency_ms),
             _tag(b.p99_kv_latency_ms, o.p99_kv_latency_ms)),
            ("Cache Hit Rate",
             f"{b.kv_hit_rate_pct:.1f}%",
             f"{o.kv_hit_rate_pct:.1f}%",
             _su(b.kv_hit_rate_pct, o.kv_hit_rate_pct, higher=True),
             _tag(b.kv_hit_rate_pct, o.kv_hit_rate_pct, higher=True)),
            ("KV Peak RAM",
             f"{b.kv_peak_ram_mb:.1f} MB",
             f"{o.kv_peak_ram_mb:.1f} MB",
             "",
             "neutral"),
        ]

        for values in rows:
            tag = values[4]
            self._tree.insert("", "end", values=values[:4], tags=(tag,))

    # -----------------------------------------------------------------------
    # Embedded chart
    # -----------------------------------------------------------------------

    def _embed_chart(self, b: emu.BenchmarkResult, o: emu.BenchmarkResult) -> None:
        """Render matplotlib figure and embed it as a tkinter canvas."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # Destroy placeholder
        for w in self._chart_placeholder.winfo_children():
            w.destroy()

        fig, axes = plt.subplots(2, 2, figsize=(7, 5),
                                 facecolor="#1a1d2e")
        fig.subplots_adjust(hspace=0.42, wspace=0.38, left=0.1,
                            right=0.97, top=0.93, bottom=0.1)

        c_base = ACCENT_RED
        c_opt  = ACCENT_BLUE
        labels = ["Baseline", "AI-SSD"]
        bcolors = [c_base, c_opt]

        def _ax_style(ax: Any) -> None:
            ax.set_facecolor(BG3)
            ax.tick_params(colors=FG2, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.title.set_color(FG)
            ax.title.set_fontsize(9)
            ax.yaxis.label.set_color(FG2)
            ax.yaxis.label.set_fontsize(8)

        # Panel A: Load Latency
        ax = axes[0, 0]
        vals = [b.dataset_latency_ms, o.dataset_latency_ms]
        bars = ax.bar(labels, vals, color=bcolors, edgecolor=BG2, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{val:.0f}ms", ha="center", fontsize=7, color=FG)
        ax.set_title("Load Latency (ms)")
        ax.set_ylabel("ms")
        _ax_style(ax)

        # Panel B: Throughput
        ax = axes[0, 1]
        vals = [b.throughput_mb_s, o.throughput_mb_s]
        bars = ax.bar(labels, vals, color=bcolors, edgecolor=BG2, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{val:.0f}", ha="center", fontsize=7, color=FG)
        ax.set_title("Throughput (MB/s)")
        ax.set_ylabel("MB/s")
        _ax_style(ax)

        # Panel C: KV Step latency line chart
        ax = axes[1, 0]
        steps = list(range(1, len(b.kv_step_latencies)+1))
        ax.plot(steps, b.kv_step_latencies,
                color=c_base, linewidth=1.5, label="Baseline", marker="o", markersize=3)
        ax.plot(steps, o.kv_step_latencies,
                color=c_opt,  linewidth=1.5, label="AI-SSD",   marker="s", markersize=3)
        ax.set_title("KV-Cache Latency / Step")
        ax.set_ylabel("ms/step")
        ax.set_xlabel("Step", fontsize=7, color=FG2)
        ax.legend(fontsize=7, facecolor=BG2, edgecolor=BORDER,
                  labelcolor=FG, framealpha=0.8)
        _ax_style(ax)

        # Panel D: Cache hit rate bar
        ax = axes[1, 1]
        hit_vals = [b.kv_hit_rate_pct, o.kv_hit_rate_pct]
        bars = ax.bar(labels, hit_vals, color=bcolors, edgecolor=BG2, width=0.5)
        for bar, val in zip(bars, hit_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", fontsize=7, color=FG)
        ax.set_ylim(0, 115)
        ax.set_title("Cache Hit Rate (%)")
        ax.set_ylabel("%")
        _ax_style(ax)

        # Embed into tkinter
        canvas = FigureCanvasTkAgg(fig, master=self._chart_placeholder)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)

        # Also save to disk
        out = emu.RESULTS_DIR / "benchmark_results.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        self._log_append(f"Chart saved -> {out.resolve()}", "metric")

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def _reset_ui(self, keep_log: bool = False) -> None:
        if not keep_log:
            self._log_txt.config(state="normal")
            self._log_txt.delete("1.0", "end")
            self._log_txt.config(state="disabled")

        # Reset phase cards
        for pid, _, _, _ in PHASES:
            self._phase_reset(pid)
        self._guide_index = -1
        self._hardware_reset()

        # Clear results table
        for row in self._tree.get_children():
            self._tree.delete(row)

        # Reset chart area
        for w in self._chart_placeholder.winfo_children():
            w.destroy()
        self._chart_lbl = tk.Label(
            self._chart_placeholder,
            text="Charts will appear here after the benchmark completes.",
            font=FONT_BODY, fg=FG2, bg=BG3,
        )
        self._chart_lbl.place(relx=0.5, rely=0.5, anchor="center")

        self._set_status("Ready", ACCENT_GRN)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
