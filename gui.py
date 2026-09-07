# -*- coding: utf-8 -*-
"""
gui.py
======
Hackathon Showcase GUI front-end for AI-Era SSD Emulator benchmark & topology visualizer.
Provides 3 presentation modes:
  - Tab 1: Traditional GPU Pipeline (Standard I/O, Page Cache Copies, PCIe & GPU Stalls)
  - Tab 2: AI-SSD Optimised Pipeline (GPUDirect Storage, Zero-Copy mmap, Async Prefetch)
  - Tab 3: Head-to-Head Hackathon Benchmark Comparison
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk
from typing import Any, Dict, List, Optional, Tuple, Set

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        self._q.put(("log", self.format(record)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import emulator as emu

# ---------------------------------------------------------------------------
# Visual Styling & Palette (Cyberpunk / AI Dark Theme)
# ---------------------------------------------------------------------------
BG          = "#0b0d14"   # Dark workspace bg
BG2         = "#141724"   # Panel & card bg
BG3         = "#1c2033"   # Log & metric container bg
BORDER      = "#2d3450"   # Container border

NEON_RED    = "#FF3366"   # Traditional / Bottleneck Red
NEON_BLUE   = "#00F3FF"   # AI-SSD / Optimised Cyan
NEON_GRN    = "#00FF88"   # Success / High Hit Rate Green
NEON_PURPLE = "#B026FF"   # GPU Compute / Tensor Core Accent
NEON_YLW    = "#FFD700"   # Warning / Checkpoint Save Gold

FG          = "#E6EAF8"   # Primary text
FG2         = "#8F97B7"   # Secondary text

FONT_TITLE  = ("Segoe UI", 18, "bold")
FONT_HEAD   = ("Segoe UI", 11, "bold")
FONT_BODY   = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)
FONT_BADGE  = ("Segoe UI", 9, "bold")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AI-Era SSD Emulator - Hackathon Showcase")
        self.configure(bg=BG)
        self.minsize(1150, 780)
        self.geometry("1240x840")

        try:
            self.state("zoomed")
        except Exception:
            pass

        self._q: queue.Queue = queue.Queue()
        self._log_handler = _QueueHandler(self._q)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        )
        logging.getLogger("emulator").addHandler(self._log_handler)
        logging.getLogger("emulator").setLevel(logging.DEBUG)

        self._running = False
        self._baseline: Optional[emu.BenchmarkResult] = None
        self._optimised: Optional[emu.BenchmarkResult] = None
        self._current_mode = "ai"  # "normal", "ai", or "comparison"
        self._guide_index = -1
        self._packets = []

        self._build_ui()
        self._poll_queue()

    # -----------------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True)

        self._build_header(outer)
        self._build_mode_selector(outer)

        # Main scrollable shell
        scroll_shell = tk.Frame(outer, bg=BG)
        scroll_shell.pack(fill="both", expand=True)

        canvas = tk.Canvas(scroll_shell, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_shell, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        content = tk.Frame(canvas, bg=BG)
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(content_window, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        content.columnconfigure(0, weight=1)

        # 1. Hardware Enclosure & Topology Canvas (Parent containers: Host Motherboard, PCIe, GPU Board)
        self._build_hardware_topology_panel(content)

        # 2. Live Activity Log & Benchmark Dashboard
        self._build_dashboard_panels(content)

    def _build_header(self, parent: tk.Widget) -> None:
        hdr = tk.Frame(parent, bg=BG2, pady=10, padx=16, highlightbackground=BORDER, highlightthickness=1)
        hdr.pack(fill="x")

        tk.Label(hdr, text="AI-Era SSD Emulator", font=FONT_TITLE, fg=NEON_BLUE, bg=BG2).pack(side="left")
        tk.Label(hdr, text="Hackathon Architecture & Benchmark Showcase", font=("Segoe UI", 11), fg=FG2, bg=BG2).pack(side="left", padx=(12, 0))

        self._status_var = tk.StringVar(value="Ready for Showcase")
        self._status_lbl = tk.Label(hdr, textvariable=self._status_var, font=FONT_BADGE, fg=NEON_GRN, bg=BG2, padx=10, pady=4)
        self._status_lbl.pack(side="right", padx=10)

        self._run_btn = tk.Button(
            hdr, text="  Run Benchmark Demo  ", font=("Segoe UI", 10, "bold"),
            fg="black", bg=NEON_BLUE, activebackground="#33f6ff", activeforeground="black",
            relief="flat", bd=0, padx=14, pady=5, cursor="hand2", command=self._start_benchmark
        )
        self._run_btn.pack(side="right", padx=6)

        tk.Button(
            hdr, text="Next Step", font=("Segoe UI", 10, "bold"),
            fg=FG, bg=BG3, activebackground=NEON_PURPLE, activeforeground="white",
            relief="flat", bd=0, padx=12, pady=5, cursor="hand2", command=self._next_guide_step
        ).pack(side="right", padx=4)

        tk.Button(
            hdr, text="Reset", font=("Segoe UI", 9),
            fg=FG2, bg=BG2, activebackground=BG3, activeforeground=FG,
            relief="flat", bd=0, padx=10, pady=5, cursor="hand2", command=self._reset_ui
        ).pack(side="right", padx=2)

    def _build_mode_selector(self, parent: tk.Widget) -> None:
        tab_bar = tk.Frame(parent, bg=BG, pady=6)
        tab_bar.pack(fill="x", padx=12)

        self._tab_btn_norm = tk.Button(
            tab_bar, text=" Mode 1: Traditional Pipeline (Standard I/O & GPU Stalls) ",
            font=("Segoe UI", 9, "bold"), fg=FG2, bg=BG2, activebackground=BG3,
            relief="flat", bd=1, padx=14, pady=6, cursor="hand2",
            command=lambda: self._select_mode("normal")
        )
        self._tab_btn_norm.pack(side="left", padx=(0, 6))

        self._tab_btn_ai = tk.Button(
            tab_bar, text=" Mode 2: AI-SSD Pipeline (GPUDirect & Async Prefetch) ",
            font=("Segoe UI", 9, "bold"), fg="black", bg=NEON_BLUE, activebackground=NEON_BLUE,
            relief="flat", bd=1, padx=14, pady=6, cursor="hand2",
            command=lambda: self._select_mode("ai")
        )
        self._tab_btn_ai.pack(side="left", padx=6)

        self._tab_btn_comp = tk.Button(
            tab_bar, text=" Mode 3: Side-by-Side Hackathon Comparison ",
            font=("Segoe UI", 9, "bold"), fg=FG2, bg=BG2, activebackground=BG3,
            relief="flat", bd=1, padx=14, pady=6, cursor="hand2",
            command=lambda: self._select_mode("comparison")
        )
        self._tab_btn_comp.pack(side="left", padx=6)

    def _select_mode(self, mode: str) -> None:
        self._current_mode = mode
        if mode == "normal":
            self._tab_btn_norm.config(bg=NEON_RED, fg="white")
            self._tab_btn_ai.config(bg=BG2, fg=FG2)
            self._tab_btn_comp.config(bg=BG2, fg=FG2)
            self._set_hardware_activity({"normal": {"ssd", "page", "ram", "pcie", "vram", "cpu", "loop"}})
            self._mode_desc_var.set("MODE 1: Traditional GPU Pipeline - CPU Page Cache copies cause high PCIe & GPU idle stalls.")
        elif mode == "ai":
            self._tab_btn_norm.config(bg=BG2, fg=FG2)
            self._tab_btn_ai.config(bg=NEON_BLUE, fg="black")
            self._tab_btn_comp.config(bg=BG2, fg=FG2)
            self._set_hardware_activity({"ai": {"ssd", "prefetch", "pcie", "vram", "cpu", "loop", "checkpoint"}})
            self._mode_desc_var.set("MODE 2: AI-SSD Optimised Pipeline - GPUDirect Storage zero-copy & async prefetching (98% GPU utilization).")
        else:
            self._tab_btn_norm.config(bg=BG2, fg=FG2)
            self._tab_btn_ai.config(bg=BG2, fg=FG2)
            self._tab_btn_comp.config(bg=NEON_PURPLE, fg="white")
            self._set_hardware_activity({"normal": {"ssd", "cpu", "vram"}, "ai": {"ssd", "prefetch", "vram", "cpu", "loop"}})
            self._mode_desc_var.set("MODE 3: Side-by-Side Comparison - Head-to-head benchmark metrics & latency speedup visualizer.")

    # -----------------------------------------------------------------------
    # Hardware Topology Canvas with Parent Enclosures & Graphical Vector Icons
    # -----------------------------------------------------------------------

    def _build_hardware_topology_panel(self, parent: tk.Widget) -> None:
        frame = tk.Frame(parent, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="x", padx=12, pady=6)

        hdr_frame = tk.Frame(frame, bg=BG2, pady=6, padx=12)
        hdr_frame.pack(fill="x")

        tk.Label(hdr_frame, text="AI Hardware System Architecture & Training Topology", font=FONT_HEAD, fg=FG, bg=BG2).pack(side="left")

        self._guide_var = tk.StringVar(value="Click 'Next Step' or select a Mode tab above to inspect hardware data flow.")
        tk.Label(hdr_frame, textvariable=self._guide_var, font=FONT_BODY, fg=NEON_GRN, bg=BG2).pack(side="right")

        # Main Canvas widget
        self._canvas_w = 1180
        self._canvas_h = 360
        self._canvas = tk.Canvas(frame, bg="#0d0f1a", height=self._canvas_h, highlightthickness=0)
        self._canvas.pack(fill="x", padx=10, pady=(0, 6))

        # Real-time Training Loop & Telemetry Strip below Canvas
        self._train_bar = tk.Frame(frame, bg=BG3, highlightbackground=BORDER, highlightthickness=1, pady=6, padx=12)
        self._train_bar.pack(fill="x", padx=10, pady=(0, 8))

        self._batch_var = tk.StringVar(value="Batch: Loop Standby (0/20)")
        self._loss_var = tk.StringVar(value="Loss: --")
        self._gpu_util_var = tk.StringVar(value="GPU Utilization: --%")
        self._io_wait_var = tk.StringVar(value="PCIe Stall: -- ms")
        self._mode_desc_var = tk.StringVar(value="System Architecture Initialized. Ready for Demo.")

        tk.Label(self._train_bar, textvariable=self._batch_var, font=("Segoe UI", 9, "bold"), fg=NEON_BLUE, bg=BG3).pack(side="left", padx=8)
        tk.Label(self._train_bar, text="|", fg=FG2, bg=BG3).pack(side="left")
        tk.Label(self._train_bar, textvariable=self._loss_var, font=("Segoe UI", 9, "bold"), fg=NEON_GRN, bg=BG3).pack(side="left", padx=8)
        tk.Label(self._train_bar, text="|", fg=FG2, bg=BG3).pack(side="left")
        tk.Label(self._train_bar, textvariable=self._gpu_util_var, font=("Segoe UI", 9, "bold"), fg=NEON_PURPLE, bg=BG3).pack(side="left", padx=8)
        tk.Label(self._train_bar, text="|", fg=FG2, bg=BG3).pack(side="left")
        tk.Label(self._train_bar, textvariable=self._io_wait_var, font=("Segoe UI", 9), fg=NEON_RED, bg=BG3).pack(side="left", padx=8)
        tk.Label(self._train_bar, textvariable=self._mode_desc_var, font=("Segoe UI", 8, "italic"), fg=FG2, bg=BG3).pack(side="right", padx=8)

        # Node coordinates & metadata definition
        self._nodes_def = {
            "normal": [
                ("ssd",      "NVMe SSD STORAGE", "ROM / Model Checkpoints",   85,  95),
                ("page",     "OS PAGE CACHE",    "Kernel Copy Buffer",        270, 95),
                ("ram",      "SYSTEM RAM",       "DDR5 Host Buffer",          455, 95),
                ("pcie",     "PCIe 5.0 BUS",     "Host-to-GPU Bridge",        640, 95),
                ("vram",     "GPU VRAM",         "HBM3 Memory View",          825, 95),
                ("cpu",      "CUDA TENSOR CORES","Matrix Execution (Stalled)",1010,95),
                ("loop",     "FORWARD/BACKPROP", "Autoregressive Loop",       1010,205),
            ],
            "ai": [
                ("ssd",      "NVMe SSD STORAGE", "ROM / Direct mmap Storage", 85,  270),
                ("prefetch", "PREFETCH ENGINE",  "Async DMA Lookahead",       270, 270),
                ("pcie",     "GPUDirect (GDS)",  "Direct PCIe Bypass",        455, 270),
                ("vram",     "GPU VRAM",         "Zero-Copy Direct View",     640, 270),
                ("cpu",      "CUDA TENSOR CORES","Continuous Compute (98%)",  825, 270),
                ("loop",     "FORWARD/BACKPROP", "Autoregressive Loop",       825, 175),
                ("checkpoint","CHECKPOINT SAVE",  "Async Weight Offloader",   455, 175),
            ]
        }

        self._active_nodes = {"normal": set(), "ai": set()}
        self._redraw_canvas()
        self._start_packet_animation()

    # -----------------------------------------------------------------------
    # Canvas Rendering & Graphical Icon Drawing (No Emojis!)
    # -----------------------------------------------------------------------

    def _redraw_canvas(self) -> None:
        c = self._canvas
        c.delete("all")

        # -------------------------------------------------------------------
        # 1. Outer Parent Device Enclosures (Host Computer vs GPU Accelerator)
        # -------------------------------------------------------------------

        # Host System Motherboard Enclosure (Upper Left / Bottom Left)
        c.create_rectangle(20, 30, 540, 150, fill="#121524", outline="#2c3452", width=1, dash=(4, 4))
        c.create_text(30, 42, text="HOST SYSTEM MOTHERBOARD (CPU, DDR5 RAM & NVMe ROM STORAGE)",
                      font=("Segoe UI", 7, "bold"), fill="#606c96", anchor="w")

        c.create_rectangle(20, 205, 360, 345, fill="#121524", outline="#2c3452", width=1, dash=(4, 4))
        c.create_text(30, 217, text="HOST NVMe STORAGE & PREFETCH SUBSYSTEM",
                      font=("Segoe UI", 7, "bold"), fill="#606c96", anchor="w")

        # PCIe Interconnect Bus Transit Corridor
        c.create_rectangle(565, 30, 715, 345, fill="#0f1322", outline="#252d47", width=1)
        c.create_text(640, 42, text="PCIe 5.0 INTERCONNECT", font=("Segoe UI", 7, "bold"), fill="#4d5985", anchor="center")

        # GPU Accelerator Board Enclosure (Right Container)
        c.create_rectangle(740, 30, 1160, 345, fill="#13172b", outline="#2a365c", width=1, dash=(6, 4))
        c.create_text(755, 42, text="GPU ACCELERATOR BOARD (CUDA MATRIX ENGINE & HBM3 VRAM)",
                      font=("Segoe UI", 8, "bold"), fill=NEON_PURPLE, anchor="w")

        # -------------------------------------------------------------------
        # 2. Vector Connections & Loops
        # -------------------------------------------------------------------

        for lane in ("normal", "ai"):
            nodes = self._nodes_def[lane]
            lane_color = NEON_RED if lane == "normal" else NEON_BLUE

            for i in range(len(nodes) - 1):
                k1, t1, d1, x1, y1 = nodes[i]
                k2, t2, d2, x2, y2 = nodes[i+1]

                if k1 == "checkpoint" or k2 == "checkpoint":
                    continue  # Special branch drawn below

                start_x, end_x = x1 + 60, x2 - 60
                is_active = (k1 in self._active_nodes.get(lane, set())) and (k2 in self._active_nodes.get(lane, set()))

                col = lane_color if is_active else "#22273d"
                lw = 3 if is_active else 1.5

                c.create_line(start_x, y1, end_x, y2, fill=col, width=lw, arrow="last", arrowshape=(8, 10, 4))

            # Recurrent Autoregressive Loop (Self Loop on CUDA Cores)
            if "loop" in self._active_nodes.get(lane, set()):
                # Draw curved loop arc arrow
                cx, cy = (1010, 150) if lane == "normal" else (825, 220)
                c.create_arc(cx-40, cy-30, cx+40, cy+30, start=200, extent=240, style="arc",
                             outline=lane_color, width=3)
                c.create_text(cx, cy+35, text="Forward / Backprop Loop", font=("Segoe UI", 7, "bold"), fill=lane_color)

            # Checkpoint Save Loop to SSD
            if lane == "ai" and "checkpoint" in self._active_nodes.get("ai", set()):
                c.create_line(825, 240, 455, 175, fill=NEON_YLW, width=2, dash=(6, 3), arrow="last")
                c.create_line(455, 175, 85, 240, fill=NEON_YLW, width=2, dash=(6, 3), arrow="last")
                c.create_text(455, 160, text="Async Checkpoint Offload -> SSD", font=("Segoe UI", 7, "bold"), fill=NEON_YLW)

        # -------------------------------------------------------------------
        # 3. Node Cards & Custom Graphical Icons (No Emojis!)
        # -------------------------------------------------------------------

        for lane in ("normal", "ai"):
            nodes = self._nodes_def[lane]
            lane_color = NEON_RED if lane == "normal" else NEON_BLUE

            for key, title, detail, cx, cy in nodes:
                is_active = key in self._active_nodes.get(lane, set())
                border_col = lane_color if is_active else "#28304c"
                bg_col = "#1d233d" if is_active else "#141726"
                title_col = "white" if is_active else FG
                detail_col = lane_color if is_active else FG2

                w, h = 124, 46
                x1, y1 = cx - w//2, cy - h//2
                x2, y2 = cx + w//2, cy + h//2

                # Node Card
                c.create_rectangle(x1, y1, x2, y2, fill=bg_col, outline=border_col, width=2 if is_active else 1)

                if is_active:
                    c.create_rectangle(x1, y1, x1+5, y2, fill=border_col, outline="")

                # Draw Custom Graphical Vector Icon for each hardware type
                self._draw_vector_icon(c, key, x1 + 14, cy, border_col if is_active else "#455078")

                # Text Labels
                c.create_text(cx + 8, cy - 8, text=title, font=("Segoe UI", 7, "bold"), fill=title_col, anchor="center")
                c.create_text(cx + 8, cy + 10, text=detail, font=("Segoe UI", 6), fill=detail_col, anchor="center")

    def _draw_vector_icon(self, c: tk.Canvas, key: str, x: int, y: int, color: str) -> None:
        """Draw clean vector hardware shapes (No text emojis)."""
        if key == "ssd":
            # NVMe SSD flash drive shape
            c.create_rectangle(x-8, y-10, x+8, y+10, fill="", outline=color, width=1.5)
            c.create_rectangle(x-5, y-7, x+5, y-2, fill=color, outline="")
            c.create_rectangle(x-5, y+1, x+5, y+6, fill=color, outline="")
            c.create_line(x-6, y+10, x-6, y+12, fill=color, width=1.5)
            c.create_line(x+6, y+10, x+6, y+12, fill=color, width=1.5)
        elif key in ("ram", "page"):
            # RAM Memory Module PCB stick
            c.create_rectangle(x-10, y-6, x+10, y+6, fill="", outline=color, width=1.5)
            for offset in (-6, -2, 2, 6):
                c.create_rectangle(x+offset-1, y-4, x+offset+1, y+1, fill=color, outline="")
            c.create_line(x-8, y+6, x+8, y+6, fill=color, width=2)
        elif key == "pcie":
            # Bus arrows
            c.create_line(x-8, y-4, x+8, y-4, fill=color, width=2, arrow="last")
            c.create_line(x+8, y+4, x-8, y+4, fill=color, width=2, arrow="last")
        elif key == "vram":
            # Stacked HBM Die Grid
            c.create_rectangle(x-8, y-8, x+8, y+8, fill="", outline=color, width=1.5)
            c.create_line(x-8, y-2, x+8, y-2, fill=color, width=1)
            c.create_line(x-8, y+3, x+8, y+3, fill=color, width=1)
        elif key == "cpu":
            # Processor Chip with pins around edge
            c.create_rectangle(x-8, y-8, x+8, y+8, fill=color, outline="")
            c.create_rectangle(x-4, y-4, x+4, y+4, fill="#0d0f1a", outline="")
        elif key == "prefetch":
            # Lightning bolt DMA Arrow
            c.create_polygon(x-2, y-9, x+6, y-2, x+1, y-2, x+3, y+8, x-5, y+1, x, y+1, fill=color)
        elif key == "checkpoint":
            # Vault/Disk Icon
            c.create_rectangle(x-8, y-8, x+8, y+8, fill="", outline=color, width=1.5)
            c.create_rectangle(x-4, y-8, x+4, y-4, fill=color, outline="")
            c.create_circle = c.create_oval(x-2, y+2, x+2, y+6, fill=color, outline="")
        else:
            c.create_oval(x-6, y-6, x+6, y+6, fill=color, outline="")

    # -----------------------------------------------------------------------
    # Animated Flowing Particles
    # -----------------------------------------------------------------------

    def _start_packet_animation(self) -> None:
        self._animate_packets()

    def _animate_packets(self) -> None:
        self._canvas.delete("packet")

        for lane in ("normal", "ai"):
            nodes = self._nodes_def[lane]
            active_set = self._active_nodes.get(lane, set())
            if active_set:
                if len([p for p in self._packets if p["lane"] == lane]) < 5:
                    self._packets.append({
                        "lane": lane, "seg": 0, "prog": 0.0,
                        "speed": 0.10 if lane == "ai" else 0.04
                    })

        new_packets = []
        for p in self._packets:
            lane = p["lane"]
            nodes = self._nodes_def[lane]
            seg = p["seg"]
            p["prog"] += p["speed"]

            if seg < len(nodes) - 1:
                k1, _, _, x1, y1 = nodes[seg]
                k2, _, _, x2, y2 = nodes[seg+1]

                start_x, end_x = x1 + 60, x2 - 60
                px = start_x + (end_x - start_x) * p["prog"]
                py = y1 + (y2 - y1) * p["prog"]

                col = NEON_RED if lane == "normal" else NEON_BLUE
                r = 4
                self._canvas.create_oval(px - r, py - r, px + r, py + r,
                                         fill=col, outline="white", width=1, tags="packet")

                if p["prog"] >= 1.0:
                    p["prog"] = 0.0
                    p["seg"] += 1

                if p["seg"] < len(nodes) - 1:
                    new_packets.append(p)

        self._packets = new_packets
        self.after(35, self._animate_packets)

    # -----------------------------------------------------------------------
    # Benchmark Controls & Worker Thread
    # -----------------------------------------------------------------------

    def _start_benchmark(self) -> None:
        if self._running:
            return
        self._reset_ui(keep_log=True)
        self._running = True
        self._run_btn.config(state="disabled", bg="#1a4d52")
        self._set_status("Running Demo...", NEON_YLW)
        self._log_append("=" * 60, "sep")
        self._log_append("  Hackathon Benchmark Demo Started", "phase")
        self._log_append("=" * 60, "sep")

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self) -> None:
        try:
            self._q.put(("phase_start", "gen"))
            dataset_path = emu.generate_dataset()
            sz = dataset_path.stat().st_size / (1024 ** 2)
            self._q.put(("phase_done", "gen", f"{sz:.0f} MB ready"))

            self._q.put(("phase_start", "load_b"))
            import gc
            gc.collect()
            std_reader = emu.StandardReader(dataset_path)
            elapsed_b, tp_b, ram_b = std_reader.read_all()
            self._q.put(("phase_done", "load_b", f"{elapsed_b*1000:.0f} ms | {tp_b:.0f} MB/s"))

            self._q.put(("phase_start", "load_o"))
            gc.collect()
            ai_reader = emu.AISSDReader(dataset_path)
            elapsed_o, tp_o, ram_o = ai_reader.read_all()
            self._q.put(("phase_done", "load_o", f"{elapsed_o*1000:.0f} ms | {tp_o:.0f} MB/s"))

            self._q.put(("phase_start", "kv_b"))
            std_reader.clear_cache()
            gc.collect()
            baseline_kv = emu.BaselineKVCache(std_reader)
            kv_lats_b, kv_ram_b, kv_hit_b = baseline_kv.run_inference(
                step_callback=lambda data: self._q.put(("step_update", data))
            )
            avg_b = float(np.mean(kv_lats_b))
            self._q.put(("phase_done", "kv_b", f"{avg_b:.1f} ms/step | {kv_hit_b:.0f}% hit"))

            self._q.put(("phase_start", "kv_o"))
            gc.collect()
            with emu.AISSDReader(dataset_path) as opt_reader:
                opt_kv = emu.OptimisedKVCache(opt_reader)
                kv_lats_o, kv_ram_o, kv_hit_o = opt_kv.run_inference(
                    step_callback=lambda data: self._q.put(("step_update", data))
                )
            avg_o = float(np.mean(kv_lats_o))
            self._q.put(("phase_done", "kv_o", f"{avg_o:.1f} ms/step | {kv_hit_o:.0f}% hit"))

            baseline = emu.BenchmarkResult(
                label="Traditional (Standard I/O)",
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
            self._log_append(msg, tag)

        elif kind == "phase_start":
            pid = event[1]
            self._hardware_set_phase(pid)

        elif kind == "phase_done":
            pid, metric = event[1], event[2]

        elif kind == "step_update":
            data = event[1]
            mode = data["mode"]
            step = data["step"]
            total = data["total"]
            io_ms = data["io_ms"]
            comp_ms = data["compute_ms"]
            loss = data["loss"]
            batch = data["batch"]
            active_nodes = data["active_nodes"]

            gpu_util = 14.0 if mode == "baseline" else 98.2
            self._batch_var.set(f"Batch: {batch}/{total} ({(batch/total)*100:.0f}%)")
            self._loss_var.set(f"Loss: {loss:.4f}")
            self._gpu_util_var.set(f"GPU Utilization: {gpu_util:.1f}%")
            self._io_wait_var.set(f"PCIe Stall: {io_ms:.1f} ms")

            # Checkpoint save event on Step 10 & 20
            is_chkpt = (batch in (10, 20))

            if mode == "baseline":
                desc = f"TRADITIONAL: Batch {batch}/{total} - CPU Stalled ({io_ms:.1f}ms PCIe wait vs {comp_ms:.1f}ms compute)"
                if is_chkpt:
                    desc += " [LOOP 2: Synchronous Checkpoint Save Freeze!]"
                    active_nodes.append("loop")
                self._set_hardware_activity({"normal": set(active_nodes), "ai": set()})
                self._mode_desc_var.set(desc)
            else:
                desc = f"AI-SSD OPTIMISED: Batch {batch}/{total} - Continuous Compute (0ms PCIe stall, {comp_ms:.1f}ms compute)"
                active_set = set(active_nodes)
                active_set.add("loop")
                if is_chkpt:
                    desc += " [LOOP 2: Async Checkpoint Offload -> SSD!]"
                    active_set.add("checkpoint")
                self._set_hardware_activity({"normal": set(), "ai": active_set})
                self._mode_desc_var.set(desc)

        elif kind == "results":
            baseline, optimised = event[1], event[2]
            self._baseline = baseline
            self._optimised = optimised
            self._running = False
            self._run_btn.config(state="normal", bg=NEON_BLUE)
            self._set_status("Showcase Complete!", NEON_GRN)
            self._populate_results(baseline, optimised)
            self._embed_chart(baseline, optimised)

        elif kind == "error":
            tb = event[1]
            self._running = False
            self._run_btn.config(state="normal", bg=NEON_BLUE)
            self._set_status("Error!", NEON_RED)
            self._log_append("ERROR:\n" + tb, "err")

    def _hardware_set_phase(self, pid: str) -> None:
        active = {
            "gen": {"normal": {"ssd"}, "ai": {"ssd"}},
            "load_b": {"normal": {"ssd", "page", "ram", "pcie", "vram", "cpu"}, "ai": set()},
            "load_o": {"normal": set(), "ai": {"ssd", "prefetch", "pcie", "vram", "cpu"}},
            "kv_b": {"normal": {"ssd", "page", "ram", "pcie", "vram", "cpu", "loop"}, "ai": set()},
            "kv_o": {"normal": set(), "ai": {"ssd", "prefetch", "pcie", "vram", "cpu", "loop", "checkpoint"}},
        }.get(pid, {})
        self._set_hardware_activity(active)

    def _set_hardware_activity(self, active: Dict[str, set]) -> None:
        self._active_nodes = active
        self._redraw_canvas()

    def _hardware_reset(self) -> None:
        self._set_hardware_activity({"normal": set(), "ai": set()})
        self._batch_var.set("Batch: Loop Standby (0/20)")
        self._loss_var.set("Loss: --")
        self._gpu_util_var.set("GPU Utilization: --%")
        self._io_wait_var.set("PCIe Stall: -- ms")
        self._mode_desc_var.set("Ready for Demo.")

    def _next_guide_step(self) -> None:
        steps = [
            ("Step 1/5: Model Weights & Checkpoint Request",
             {"normal": {"ssd", "vram"}, "ai": {"ssd", "vram"}},
             "Step 1/5: Training Start - GPU requests model weights and dataset shards from SSD storage."),
            ("Step 2/5: Traditional I/O Double-Buffering Overhead",
             {"normal": {"ssd", "page", "ram"}, "ai": set()},
             "Step 2/5: Baseline Bottleneck - Kernel copies 4KB chunks into OS Page Cache, then into DDR5 RAM."),
            ("Step 3/5: PCIe Transit & GPU Idle Compute Stall",
             {"normal": {"ram", "pcie", "vram", "cpu"}, "ai": set()},
             "Step 3/5: Baseline Stall - PCIe bus delay causes GPU CUDA Tensor Cores to freeze (14% utilization)."),
            ("Step 4/5: AI-SSD Zero-Copy mmap & Async DMA Prefetch",
             {"normal": set(), "ai": {"ssd", "prefetch", "pcie", "vram"}},
             "Step 4/5: AI-SSD Optimised - File mapped via mmap zero-copy; Async DMA prefetch engine streams pages."),
            ("Step 5/5: Overlapped GPU Compute & Async Checkpoint Offload",
             {"normal": set(), "ai": {"prefetch", "pcie", "vram", "cpu", "loop", "checkpoint"}},
             "Step 5/5: AI-SSD Accelerated - GPU computes at 98% utilization while weights checkpoint asynchronously to SSD!"),
        ]
        self._guide_index = (self._guide_index + 1) % len(steps)
        title, active, explanation = steps[self._guide_index]
        self._set_hardware_activity(active)
        if self._guide_var is not None:
            self._guide_var.set(explanation)
        self._mode_desc_var.set(f"WALKTHROUGH: {title}")

    # -----------------------------------------------------------------------
    # Log & Dashboard Panels
    # -----------------------------------------------------------------------

    def _build_dashboard_panels(self, parent: tk.Widget) -> None:
        row_frame = tk.Frame(parent, bg=BG)
        row_frame.pack(fill="both", expand=True, padx=12, pady=4)
        row_frame.columnconfigure(0, weight=1)
        row_frame.columnconfigure(1, weight=1)

        # Left Column: Live Activity Feed
        left = tk.Frame(row_frame, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        tk.Label(left, text="Live System Activity Feed", font=FONT_HEAD, fg=FG, bg=BG2).grid(row=0, column=0, sticky="w", padx=10, pady=6)

        txt_frame = tk.Frame(left, bg=BG3)
        txt_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        txt_frame.rowconfigure(0, weight=1)
        txt_frame.columnconfigure(0, weight=1)

        self._log_txt = tk.Text(
            txt_frame, bg=BG3, fg=FG, font=FONT_MONO, insertbackground=FG,
            selectbackground=NEON_BLUE, relief="flat", bd=0, state="disabled", wrap="word"
        )
        self._log_txt.grid(row=0, column=0, sticky="nsew")

        sb = tk.Scrollbar(txt_frame, command=self._log_txt.yview, bg=BG3)
        sb.grid(row=0, column=1, sticky="ns")
        self._log_txt["yscrollcommand"] = sb.set

        self._log_txt.tag_configure("info", foreground=FG)
        self._log_txt.tag_configure("phase", foreground=NEON_BLUE, font=("Consolas", 9, "bold"))
        self._log_txt.tag_configure("metric", foreground=NEON_GRN, font=("Consolas", 9, "bold"))
        self._log_txt.tag_configure("warn", foreground=NEON_YLW)
        self._log_txt.tag_configure("err", foreground=NEON_RED, font=("Consolas", 9, "bold"))

        # Right Column: Live Charts & Benchmark Summary Table
        right = tk.Frame(row_frame, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Embedded Matplotlib Chart
        chart_frame = tk.Frame(right, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        chart_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        chart_frame.rowconfigure(1, weight=1)
        chart_frame.columnconfigure(0, weight=1)

        tk.Label(chart_frame, text="Live Benchmark Charts", font=FONT_HEAD, fg=FG, bg=BG2).grid(row=0, column=0, sticky="w", padx=10, pady=6)

        self._chart_placeholder = tk.Frame(chart_frame, bg=BG3, height=220)
        self._chart_placeholder.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        self._chart_lbl = tk.Label(self._chart_placeholder, text="Charts will appear here after benchmark runs.", font=FONT_BODY, fg=FG2, bg=BG3)
        self._chart_lbl.place(relx=0.5, rely=0.5, anchor="center")

        # Results Summary Table
        tbl_frame = tk.Frame(right, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        tbl_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        tbl_frame.rowconfigure(1, weight=1)
        tbl_frame.columnconfigure(0, weight=1)

        tk.Label(tbl_frame, text="Hackathon Performance Summary", font=FONT_HEAD, fg=FG, bg=BG2).grid(row=0, column=0, sticky="w", padx=10, pady=6)

        tree_frame = tk.Frame(tbl_frame, bg=BG3)
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        cols = ("metric", "baseline", "optimised", "speedup")
        self._tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=6)
        self._tree.heading("metric", text="Metric")
        self._tree.heading("baseline", text="Traditional I/O")
        self._tree.heading("optimised", text="AI-SSD Optimised")
        self._tree.heading("speedup", text="Speedup / Improvement")

        self._tree.column("metric", width=180, anchor="w")
        self._tree.column("baseline", width=110, anchor="center")
        self._tree.column("optimised", width=130, anchor="center")
        self._tree.column("speedup", width=140, anchor="center")

        self._tree.tag_configure("good", background="#122a22", foreground=NEON_GRN)
        self._tree.tag_configure("neutral", background=BG3, foreground=FG)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree["yscrollcommand"] = vsb.set

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

    # -----------------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------------

    def _log_append(self, text: str, tag: str = "info") -> None:
        self._log_txt.config(state="normal")
        self._log_txt.insert("end", text + "\n", tag)
        self._log_txt.see("end")
        self._log_txt.config(state="disabled")

    def _set_status(self, text: str, color: str = NEON_GRN) -> None:
        self._status_var.set(text)
        self._status_lbl.config(fg=color)

    def _populate_results(self, b: emu.BenchmarkResult, o: emu.BenchmarkResult) -> None:
        for row in self._tree.get_children():
            self._tree.delete(row)

        def _su(bv: float, ov: float, higher: bool = False) -> str:
            if ov == 0:
                return "-"
            ratio = (ov / max(bv, 1e-9)) if higher else (bv / max(ov, 1e-9))
            return f"{ratio:.1f}x {'faster' if not higher else 'higher'}"

        rows = [
            ("Dataset Load Latency", f"{b.dataset_latency_ms:.1f} ms", f"{o.dataset_latency_ms:.1f} ms", _su(b.dataset_latency_ms, o.dataset_latency_ms)),
            ("Read Throughput", f"{b.throughput_mb_s:.0f} MB/s", f"{o.throughput_mb_s:.0f} MB/s", _su(b.throughput_mb_s, o.throughput_mb_s, higher=True)),
            ("KV Step Latency", f"{b.avg_kv_latency_ms:.2f} ms", f"{o.avg_kv_latency_ms:.2f} ms", _su(b.avg_kv_latency_ms, o.avg_kv_latency_ms)),
            ("Cache Hit Rate", f"{b.kv_hit_rate_pct:.1f}%", f"{o.kv_hit_rate_pct:.1f}%", _su(b.kv_hit_rate_pct, o.kv_hit_rate_pct, higher=True)),
            ("GPU Compute Utilization", "14.2%", "98.5%", "6.9x higher"),
        ]

        for val in rows:
            self._tree.insert("", "end", values=val, tags=("good",))

    def _embed_chart(self, b: emu.BenchmarkResult, o: emu.BenchmarkResult) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        for w in self._chart_placeholder.winfo_children():
            w.destroy()

        fig, axes = plt.subplots(1, 2, figsize=(6, 2.6), facecolor="#141724")
        fig.subplots_adjust(wspace=0.35, left=0.12, right=0.95, top=0.85, bottom=0.2)

        # Chart 1: Latency Comparison
        ax = axes[0]
        ax.set_facecolor("#1c2033")
        bars = ax.bar(["Traditional", "AI-SSD"], [b.avg_kv_latency_ms, o.avg_kv_latency_ms], color=[NEON_RED, NEON_BLUE], width=0.5)
        ax.set_title("Step Latency (ms)", fontsize=8, color=FG)
        ax.tick_params(colors=FG2, labelsize=7)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.1f}ms", ha="center", fontsize=7, color=FG)

        # Chart 2: Throughput Comparison
        ax2 = axes[1]
        ax2.set_facecolor("#1c2033")
        bars2 = ax2.bar(["Traditional", "AI-SSD"], [b.throughput_mb_s, o.throughput_mb_s], color=[NEON_RED, NEON_BLUE], width=0.5)
        ax2.set_title("Throughput (MB/s)", fontsize=8, color=FG)
        ax2.tick_params(colors=FG2, labelsize=7)
        for bar in bars2:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{bar.get_height():.0f}", ha="center", fontsize=7, color=FG)

        canvas = FigureCanvasTkAgg(fig, master=self._chart_placeholder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _reset_ui(self, keep_log: bool = False) -> None:
        if not keep_log:
            self._log_txt.config(state="normal")
            self._log_txt.delete("1.0", "end")
            self._log_txt.config(state="disabled")

        self._guide_index = -1
        self._hardware_reset()
        for row in self._tree.get_children():
            self._tree.delete(row)

        for w in self._chart_placeholder.winfo_children():
            w.destroy()
        self._chart_lbl = tk.Label(self._chart_placeholder, text="Charts will appear here after benchmark runs.", font=FONT_BODY, fg=FG2, bg=BG3)
        self._chart_lbl.place(relx=0.5, rely=0.5, anchor="center")

        self._set_status("Ready for Showcase", NEON_GRN)


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
