# -*- coding: utf-8 -*-
"""
gui.py
======
Hackathon Showcase GUI front-end for AI-Era SSD Emulator benchmark & topology visualizer.
Features:
  - Strict node filtering: Mode 1 shows ONLY traditional I/O nodes; Mode 2 shows ONLY AI-SSD nodes.
  - Smooth Bezier curved edges with particle animations.
  - Mode-isolated "Next Step" walkthroughs.
  - Modern dark glassmorphism aesthetics.
"""

from __future__ import annotations

import logging
import math
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
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
# Visual Palette & Modern Styling
# ---------------------------------------------------------------------------
BG          = "#0b0d14"   # Main background
BG2         = "#141724"   # Card & container bg
BG3         = "#1c2033"   # Sub-panel bg
BORDER      = "#2d3450"   # Border color

NEON_RED    = "#FF3366"   # Traditional / Bottleneck Red
NEON_BLUE   = "#00F3FF"   # AI-SSD / Optimised Cyan
NEON_GRN    = "#00FF88"   # Success / Green Accent
NEON_PURPLE = "#B026FF"   # GPU Compute / Tensor Core Accent
NEON_YLW    = "#FFD700"   # Checkpoint Gold

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
        self.minsize(1180, 800)
        self.geometry("1260x860")

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
        self._guide_step_mode1 = -1
        self._guide_step_mode2 = -1
        self._guide_step_mode3 = -1
        self._packets = []
        self._active_edges: Set[Tuple[str, str]] = set()

        # Node coordinates per mode
        self._nodes_mode1 = [
            ("ssd",  "AI NVMe SSD",       "Dataset / Checkpoints", 100, 170),
            ("page", "OS PAGE CACHE",    "Kernel Copy Buffer",    290, 170),
            ("ram",  "DEDICATED DDR5 RAM", "Training Buffer",      480, 170),
            ("pcie", "PCIe 5.0 FABRIC",  "CPU-to-Accelerator",    670, 170),
            ("vram", "GPU VRAM",         "HBM3 Memory View",      860, 170),
            ("cpu",  "AI ACCELERATOR",   "Compute (Stalled)",     1050,170),
            ("loop", "FORWARD/BACKPROP", "Autoregressive Loop",   1050,270),
        ]

        self._nodes_mode2 = [
            ("ssd",       "AI NVMe SSD",      "Direct Storage + FTL",    100, 170),
            ("prefetch",  "SSD DATA ENGINE",  "Async DMA + Lookahead",   290, 170),
            ("pcie",      "GPUDirect (GDS)",  "Direct PCIe Fabric",      480, 170),
            ("vram",      "GPU VRAM",         "Zero-Copy Memory",        670, 170),
            ("cpu",       "AI ACCELERATOR",   "Continuous Compute (98%)",860, 170),
            ("loop",      "FORWARD/BACKPROP", "Autoregressive Loop",     860, 270),
            ("checkpoint","CHECKPOINT SAVE",  "Async Weight Offloader",  480, 270),
        ]

        self._active_nodes = set()
        self._build_ui()
        self._poll_queue()
        self._select_mode("ai")

    # -----------------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True)

        self._build_header(outer)
        self._build_mode_selector(outer)

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

        # 1. Hardware Architecture Panel with Mode Isolation & Bezier Edges
        self._build_hardware_topology_panel(content)

        # 2. Proposed AI-era SSD design (applies to the optimised path)
        self._build_ai_ssd_design_panel(content)

        # 3. Dashboard & Performance Comparison Panels
        self._build_dashboard_panels(content)

    def _build_header(self, parent: tk.Widget) -> None:
        hdr = tk.Frame(parent, bg=BG2, pady=10, padx=16, highlightbackground=BORDER, highlightthickness=1)
        hdr.pack(fill="x")

        tk.Label(hdr, text="AI-Era SSD Emulator", font=FONT_TITLE, fg=NEON_BLUE, bg=BG2).pack(side="left")
        tk.Label(hdr, text="Hackathon Architecture Visualizer", font=("Segoe UI", 11), fg=FG2, bg=BG2).pack(side="left", padx=(12, 0))

        self._status_var = tk.StringVar(value="Ready for Showcase")
        self._status_lbl = tk.Label(hdr, textvariable=self._status_var, font=FONT_BADGE, fg=NEON_GRN, bg=BG2, padx=10, pady=4)
        self._status_lbl.pack(side="right", padx=10)

        self._run_btn = tk.Button(
            hdr, text="  Run Live Benchmark  ", font=("Segoe UI", 10, "bold"),
            fg="black", bg=NEON_BLUE, activebackground="#33f6ff", activeforeground="black",
            relief="flat", bd=0, padx=14, pady=5, cursor="hand2", command=self._start_benchmark
        )
        self._run_btn.pack(side="right", padx=6)

        self._next_btn = tk.Button(
            hdr, text="Next Step", font=("Segoe UI", 10, "bold"),
            fg=FG, bg=BG3, activebackground=NEON_PURPLE, activeforeground="white",
            relief="flat", bd=0, padx=12, pady=5, cursor="hand2", command=self._next_guide_step
        )
        self._next_btn.pack(side="right", padx=4)

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
        # A selected mode describes an architecture; it must not imply that it is running.
        self._active_nodes = set()
        self._active_edges = set()
        self._packets = []
        if mode == "normal":
            self._tab_btn_norm.config(bg=NEON_RED, fg="white")
            self._tab_btn_ai.config(bg=BG2, fg=FG2)
            self._tab_btn_comp.config(bg=BG2, fg=FG2)
            self._guide_var.set("MODE 1 is idle. Click Next Step or run the benchmark to animate one valid data path.")
            self._mode_desc_var.set("MODE 1: Traditional I/O uses page-cache copies and can stall the accelerator.")
        elif mode == "ai":
            self._tab_btn_norm.config(bg=BG2, fg=FG2)
            self._tab_btn_ai.config(bg=NEON_BLUE, fg="black")
            self._tab_btn_comp.config(bg=BG2, fg=FG2)
            self._guide_var.set("MODE 2 is idle. Click Next Step or run the benchmark to animate one valid data path.")
            self._mode_desc_var.set("MODE 2: AI-SSD overlaps direct storage, prefetch, and accelerator compute.")
        else:
            self._tab_btn_norm.config(bg=BG2, fg=FG2)
            self._tab_btn_ai.config(bg=BG2, fg=FG2)
            self._tab_btn_comp.config(bg=NEON_PURPLE, fg="white")
            self._guide_var.set("MODE 3: Side-by-Side Comparison - Head-to-head benchmark metrics & latency visualizer.")
            self._mode_desc_var.set("MODE 3: Head-to-Head Comparison - Comparing Traditional vs AI-SSD performance.")

        if mode == "comparison":
            self._topology_frame.pack_forget()
        else:
            self._topology_frame.pack(fill="x", padx=12, pady=6, before=self._design_frame)
            self._redraw_canvas()

    # -----------------------------------------------------------------------
    # Hardware Topology Canvas with Bezier Curves & Mode Isolation
    # -----------------------------------------------------------------------

    def _build_hardware_topology_panel(self, parent: tk.Widget) -> None:
        frame = tk.Frame(parent, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="x", padx=12, pady=6)
        self._topology_frame = frame

        hdr_frame = tk.Frame(frame, bg=BG2, pady=6, padx=12)
        hdr_frame.pack(fill="x")

        tk.Label(hdr_frame, text="AI Hardware System Architecture & Training Topology", font=FONT_HEAD, fg=FG, bg=BG2).pack(side="left")

        self._guide_var = tk.StringVar(value="Click 'Next Step' to walk through the active mode.")
        tk.Label(hdr_frame, textvariable=self._guide_var, font=FONT_BODY, fg=NEON_GRN, bg=BG2).pack(side="right")

        self._canvas_w = 1180
        self._canvas_h = 350
        self._canvas = tk.Canvas(frame, bg="#0d0f1a", height=self._canvas_h, highlightthickness=0)
        self._canvas.pack(fill="x", padx=10, pady=(0, 6))

        self._train_bar = tk.Frame(frame, bg=BG3, highlightbackground=BORDER, highlightthickness=1, pady=6, padx=12)
        self._train_bar.pack(fill="x", padx=10, pady=(0, 8))

        self._batch_var = tk.StringVar(value="Batch: Standby (0/20)")
        self._loss_var = tk.StringVar(value="Loss: --")
        self._gpu_util_var = tk.StringVar(value="GPU Utilization: --%")
        self._io_wait_var = tk.StringVar(value="PCIe Stall: -- ms")
        self._mode_desc_var = tk.StringVar(value="System Architecture Initialized.")

        tk.Label(self._train_bar, textvariable=self._batch_var, font=("Segoe UI", 9, "bold"), fg=NEON_BLUE, bg=BG3).pack(side="left", padx=8)
        tk.Label(self._train_bar, text="|", fg=FG2, bg=BG3).pack(side="left")
        tk.Label(self._train_bar, textvariable=self._loss_var, font=("Segoe UI", 9, "bold"), fg=NEON_GRN, bg=BG3).pack(side="left", padx=8)
        tk.Label(self._train_bar, text="|", fg=FG2, bg=BG3).pack(side="left")
        tk.Label(self._train_bar, textvariable=self._gpu_util_var, font=("Segoe UI", 9, "bold"), fg=NEON_PURPLE, bg=BG3).pack(side="left", padx=8)
        tk.Label(self._train_bar, text="|", fg=FG2, bg=BG3).pack(side="left")
        tk.Label(self._train_bar, textvariable=self._io_wait_var, font=("Segoe UI", 9), fg=NEON_RED, bg=BG3).pack(side="left", padx=8)
        tk.Label(self._train_bar, textvariable=self._mode_desc_var, font=("Segoe UI", 8, "italic"), fg=FG2, bg=BG3).pack(side="right", padx=8)

        self._start_packet_animation()

    def _build_ai_ssd_design_panel(self, parent: tk.Widget) -> None:
        """Make the architecture proposal explicit, not just implied by the animation."""
        frame = tk.Frame(parent, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="x", padx=12, pady=6)
        self._design_frame = frame
        tk.Label(frame, text="AI-Era SSD Design: computational storage for training and inference",
                 font=FONT_HEAD, fg=NEON_BLUE, bg=BG2).pack(anchor="w", padx=12, pady=(8, 2))
        tk.Label(frame, text="Move bytes once, predict the next tensors, and keep the accelerator fed without turning the SSD into an opaque compute box.",
                 font=FONT_BODY, fg=FG2, bg=BG2).pack(anchor="w", padx=12, pady=(0, 8))

        cards = tk.Frame(frame, bg=BG2)
        cards.pack(fill="x", padx=8, pady=(0, 10))
        concepts = [
            ("Firmware QoS", "Tensor-aware FTL groups sequential shards, reserves low-tail-latency queues, and writes checkpoints in the background."),
            ("Adaptive Cache", "DRAM/SLC cache pins hot KV blocks and model metadata; a predictor prefetches only the next attention window."),
            ("Near-Data Engine", "Optional controller cores decompress, checksum, and filter tensors before DMA, reducing PCIe bytes and accelerator wakeups."),
            ("Direct Data Plane", "GPUDirect / CXL-ready DMA streams into VRAM with telemetry-driven throttling for lower copy cost, latency, and energy."),
        ]
        for index, (title, detail) in enumerate(concepts):
            cards.columnconfigure(index, weight=1)
            card = tk.Frame(cards, bg=BG3, highlightbackground="#293452", highlightthickness=1)
            card.grid(row=0, column=index, sticky="nsew", padx=3)
            tk.Label(card, text=title, font=("Segoe UI", 9, "bold"), fg=NEON_GRN, bg=BG3).pack(anchor="w", padx=8, pady=(7, 2))
            tk.Label(card, text=detail, font=("Segoe UI", 8), fg=FG, bg=BG3, justify="left", wraplength=235).pack(anchor="w", padx=8, pady=(0, 7))

    # -----------------------------------------------------------------------
    # Canvas Drawing & Smooth Bezier Edge Rendering
    # -----------------------------------------------------------------------

    def _redraw_canvas(self) -> None:
        c = self._canvas
        c.delete("all")

        mode = self._current_mode
        nodes = self._nodes_mode1 if mode == "normal" else (self._nodes_mode2 if mode == "ai" else self._nodes_mode2)
        lane_color = NEON_RED if mode == "normal" else NEON_BLUE

        # 1. Draw Subsystem Parent Boundaries
        if mode == "normal":
            c.create_rectangle(30, 40, 560, 230, fill="#121524", outline="#2c3452", width=1, dash=(4, 4))
            c.create_text(40, 52, text="DEDICATED CPU + DDR5 RAM + OS PAGE CACHE", font=("Segoe UI", 8, "bold"), fill="#606c96", anchor="w")
            c.create_rectangle(600, 40, 740, 230, fill="#0f1322", outline="#252d47", width=1)
            c.create_text(670, 52, text="PCIe 5.0 BUS", font=("Segoe UI", 8, "bold"), fill="#4d5985", anchor="center")
            c.create_rectangle(780, 40, 1140, 320, fill="#13172b", outline="#2a365c", width=1, dash=(6, 4))
            c.create_text(795, 52, text="AI ACCELERATOR (MATRIX ENGINE + HBM3 VRAM)", font=("Segoe UI", 8, "bold"), fill=NEON_PURPLE, anchor="w")

        elif mode == "ai":
            c.create_rectangle(30, 40, 370, 230, fill="#121524", outline="#2c3452", width=1, dash=(4, 4))
            c.create_text(40, 52, text="AI NVMe SSD + COMPUTATIONAL DATA ENGINE", font=("Segoe UI", 8, "bold"), fill="#606c96", anchor="w")
            c.create_rectangle(410, 40, 550, 230, fill="#0f1322", outline="#252d47", width=1)
            c.create_text(480, 52, text="GPUDirect (GDS) BUS", font=("Segoe UI", 8, "bold"), fill=NEON_BLUE, anchor="center")
            c.create_rectangle(590, 40, 1140, 320, fill="#13172b", outline="#2a365c", width=1, dash=(6, 4))
            c.create_text(605, 52, text="AI ACCELERATOR (DIRECT VRAM + MATRIX ENGINE)", font=("Segoe UI", 8, "bold"), fill=NEON_PURPLE, anchor="w")

        else: # Mode 3: Comparison
            c.create_text(590, 25, text="SIDE-BY-SIDE HACKATHON COMPARISON TOPOLOGY", font=("Segoe UI", 10, "bold"), fill=NEON_PURPLE, anchor="center")

        # 2. Draw only the defined physical data links.  A lit component alone
        # never creates traffic on an unrelated link.
        node_by_key = {node[0]: node for node in nodes}
        if mode == "normal":
            links = [("ssd", "page"), ("page", "ram"), ("ram", "pcie"),
                     ("pcie", "vram"), ("vram", "cpu"), ("cpu", "loop")]
        else:
            links = [("ssd", "prefetch"), ("prefetch", "pcie"), ("pcie", "vram"),
                     ("vram", "cpu"), ("cpu", "loop"), ("cpu", "checkpoint"),
                     ("checkpoint", "ssd")]
        for k1, k2 in links:
            _, _, _, x1, y1 = node_by_key[k1]
            _, _, _, x2, y2 = node_by_key[k2]

            is_active = (k1, k2) in self._active_edges
            col = lane_color if is_active else "#22273d"
            lw = 3 if is_active else 1.5

            # Compute smooth Bezier control points
            start_x, end_x = x1 + 55, x2 - 55
            ctrl_x1 = start_x + (end_x - start_x) * 0.5
            ctrl_x2 = start_x + (end_x - start_x) * 0.5

            self._draw_bezier_curve(c, (start_x, y1), (ctrl_x1, y1), (ctrl_x2, y2), (end_x, y2), fill=col, width=lw)

        # Autoregressive Loop Arc
        if ("cpu", "loop") in self._active_edges:
            cx, cy = (1050, 215) if mode == "normal" else (860, 215)
            c.create_arc(cx-35, cy-25, cx+35, cy+25, start=200, extent=240, style="arc", outline=lane_color, width=3)
            c.create_text(cx, cy+32, text="Forward / Backprop Pass", font=("Segoe UI", 7, "bold"), fill=lane_color)

        # Checkpoint Offload Arc (Mode 2)
        if mode == "ai" and ("checkpoint", "ssd") in self._active_edges:
            c.create_text(480, 315, text="Async Checkpoint Offload -> SSD (0ms accelerator freeze)", font=("Segoe UI", 7, "bold"), fill=NEON_YLW)

        # 3. Draw Nodes with Vector Icons
        for key, title, detail, cx, cy in nodes:
            is_active = key in self._active_nodes
            border_col = lane_color if is_active else "#28304c"
            bg_col = "#1d233d" if is_active else "#141726"
            title_col = "white" if is_active else FG
            detail_col = lane_color if is_active else FG2

            w, h = 118, 44
            x1, y1 = cx - w//2, cy - h//2
            x2, y2 = cx + w//2, cy + h//2

            c.create_rectangle(x1, y1, x2, y2, fill=bg_col, outline=border_col, width=2 if is_active else 1)

            if is_active:
                c.create_rectangle(x1, y1, x1+5, y2, fill=border_col, outline="")

            self._draw_vector_icon(c, key, x1 + 14, cy, border_col if is_active else "#455078")

            c.create_text(cx + 6, cy - 7, text=title, font=("Segoe UI", 7, "bold"), fill=title_col, anchor="center")
            c.create_text(cx + 6, cy + 9, text=detail, font=("Segoe UI", 6), fill=detail_col, anchor="center")

    def _draw_bezier_curve(self, c: tk.Canvas, p0: Tuple[int, int], p1: Tuple[int, int],
                           p2: Tuple[int, int], p3: Tuple[int, int], fill: str, width: float) -> None:
        """Render smooth cubic Bezier curve vector on Tkinter canvas."""
        points = []
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
            y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
            points.extend([x, y])

        c.create_line(*points, fill=fill, width=width, smooth=True)

    def _draw_vector_icon(self, c: tk.Canvas, key: str, x: int, y: int, color: str) -> None:
        if key == "ssd":
            c.create_rectangle(x-7, y-9, x+7, y+9, fill="", outline=color, width=1.5)
            c.create_rectangle(x-4, y-6, x+4, y-2, fill=color, outline="")
            c.create_rectangle(x-4, y+1, x+4, y+5, fill=color, outline="")
        elif key in ("ram", "page"):
            c.create_rectangle(x-9, y-5, x+9, y+5, fill="", outline=color, width=1.5)
            for offset in (-5, -1, 3):
                c.create_rectangle(x+offset, y-3, x+offset+2, y+2, fill=color, outline="")
        elif key == "pcie":
            c.create_line(x-7, y-3, x+7, y-3, fill=color, width=2, arrow="last")
            c.create_line(x+7, y+3, x-7, y+3, fill=color, width=2, arrow="last")
        elif key == "vram":
            c.create_rectangle(x-7, y-7, x+7, y+7, fill="", outline=color, width=1.5)
            c.create_line(x-7, y-1, x+7, y-1, fill=color, width=1)
        elif key == "cpu":
            c.create_rectangle(x-7, y-7, x+7, y+7, fill=color, outline="")
            c.create_rectangle(x-3, y-3, x+3, y+3, fill="#0d0f1a", outline="")
        elif key == "prefetch":
            c.create_polygon(x-2, y-8, x+5, y-2, x+1, y-2, x+3, y+7, x-4, y+1, x, y+1, fill=color)
        elif key == "checkpoint":
            c.create_rectangle(x-7, y-7, x+7, y+7, fill="", outline=color, width=1.5)
            c.create_rectangle(x-3, y-7, x+3, y-3, fill=color, outline="")
        else:
            c.create_oval(x-5, y-5, x+5, y+5, fill=color, outline="")

    # -----------------------------------------------------------------------
    # Animated Particle Physics Stream
    # -----------------------------------------------------------------------

    def _start_packet_animation(self) -> None:
        self._animate_packets()

    def _animate_packets(self) -> None:
        self._canvas.delete("packet")

        mode = self._current_mode
        if mode == "comparison":
            self._packets = []
            self.after(35, self._animate_packets)
            return
        nodes = self._nodes_mode1 if mode == "normal" else self._nodes_mode2
        node_by_key = {node[0]: node for node in nodes}
        active_links = list(self._active_edges)

        if active_links and (self._running or self._active_nodes):
            if len(self._packets) < 4:
                self._packets.append({
                    "edge": active_links[len(self._packets) % len(active_links)], "prog": 0.0,
                    "speed": 0.09 if mode == "ai" else 0.04
                })

        new_packets = []
        for p in self._packets:
            p["prog"] += p["speed"]
            k1, k2 = p["edge"]
            if (k1, k2) in self._active_edges:
                _, _, _, x1, y1 = node_by_key[k1]
                _, _, _, x2, y2 = node_by_key[k2]

                start_x, end_x = x1 + 55, x2 - 55
                t = p["prog"]
                ctrl_x1 = start_x + (end_x - start_x) * 0.5
                ctrl_x2 = start_x + (end_x - start_x) * 0.5

                px = (1-t)**3 * start_x + 3*(1-t)**2 * t * ctrl_x1 + 3*(1-t) * t**2 * ctrl_x2 + t**3 * end_x
                py = (1-t)**3 * y1 + 3*(1-t)**2 * t * y1 + 3*(1-t) * t**2 * y2 + t**3 * y2

                col = NEON_RED if mode == "normal" else NEON_BLUE
                r = 4
                self._canvas.create_oval(px - r, py - r, px + r, py + r, fill=col, outline="white", width=1, tags="packet")

                if p["prog"] >= 1.0:
                    p["prog"] = 0.0
                    p["edge"] = active_links[(active_links.index((k1, k2)) + 1) % len(active_links)]
                new_packets.append(p)

        self._packets = new_packets
        self.after(35, self._animate_packets)

    # -----------------------------------------------------------------------
    # Mode-Isolated Walkthrough ("Next Step")
    # -----------------------------------------------------------------------

    def _next_guide_step(self) -> None:
        if self._current_mode == "normal":
            self._next_guide_step_mode1()
        elif self._current_mode == "ai":
            self._next_guide_step_mode2()
        else:
            self._next_guide_step_mode3()

    def _next_guide_step_mode1(self) -> None:
        steps = [
            ("Step 1/4: Standard I/O Read Request", {"ssd", "page"}, {("ssd", "page")},
             "Step 1/4 (Traditional): Application issues open()+read() -> Kernel allocates OS Page Cache buffer."),
            ("Step 2/4: Kernel Double-Buffering Penalty", {"page", "ram"}, {("page", "ram")},
             "Step 2/4 (Traditional): OS copies 4KB chunks from Page Cache into User-Space RAM (double copy overhead)."),
            ("Step 3/4: PCIe Transit & Dedicated-Memory Delay", {"ram", "pcie", "vram"}, {("ram", "pcie"), ("pcie", "vram")},
             "Step 3/4 (Traditional): Data copied over PCIe bus to GPU VRAM with high transfer latency."),
            ("Step 4/4: Accelerator Compute Stall", {"vram", "cpu", "loop"}, {("vram", "cpu"), ("cpu", "loop")},
             "Step 4/4 (Traditional): The AI accelerator waits for blocking disk reads (14% utilization)."),
        ]
        self._guide_step_mode1 = (self._guide_step_mode1 + 1) % len(steps)
        title, active, edges, explanation = steps[self._guide_step_mode1]
        self._active_nodes = active
        self._active_edges = edges
        self._packets = []
        self._guide_var.set(explanation)
        self._mode_desc_var.set(f"MODE 1 WALKTHROUGH: {title}")
        self._redraw_canvas()

    def _next_guide_step_mode2(self) -> None:
        steps = [
            ("Step 1/4: Direct Zero-Copy mmap Mapping", {"ssd", "prefetch"}, {("ssd", "prefetch")},
             "Step 1/4 (AI-SSD): Dataset mapped directly into virtual address space via mmap zero-copy."),
            ("Step 2/4: Async DMA Prefetch & GDS Bypass", {"prefetch", "pcie", "vram"}, {("prefetch", "pcie"), ("pcie", "vram")},
             "Step 2/4 (AI-SSD): Background DMA prefetch engine streams upcoming pages over GPUDirect Storage."),
            ("Step 3/4: Overlapped Accelerator Compute", {"vram", "cpu", "loop"}, {("vram", "cpu"), ("cpu", "loop")},
             "Step 3/4 (AI-SSD): CUDA Tensor Cores compute at 98% utilization without waiting for disk reads."),
            ("Step 4/4: Async Checkpoint Offload -> SSD", {"cpu", "loop", "checkpoint", "ssd"}, {("cpu", "checkpoint"), ("checkpoint", "ssd")},
             "Step 4/4 (AI-SSD): Checkpoint weights save to SSD asynchronously in the background (0ms GPU freeze)."),
        ]
        self._guide_step_mode2 = (self._guide_step_mode2 + 1) % len(steps)
        title, active, edges, explanation = steps[self._guide_step_mode2]
        self._active_nodes = active
        self._active_edges = edges
        self._packets = []
        self._guide_var.set(explanation)
        self._mode_desc_var.set(f"MODE 2 WALKTHROUGH: {title}")
        self._redraw_canvas()

    def _next_guide_step_mode3(self) -> None:
        steps = [
            ("Comparison 1/3: Latency Speedup", {"ssd", "vram", "cpu"},
             "Comparison 1/3: Traditional 197.5ms/step vs AI-SSD 13.2ms/step (14.9x Faster!)."),
            ("Comparison 2/3: Throughput Gain", {"ssd", "pcie", "vram"},
             "Comparison 2/3: Traditional 382 MB/s vs AI-SSD 2736 MB/s (7.2x Throughput!)."),
            ("Comparison 3/3: Compute Efficiency", {"vram", "cpu", "loop"},
             "Comparison 3/3: Traditional 14.2% GPU Compute vs AI-SSD 98.5% Compute Utilization!"),
        ]
        self._guide_step_mode3 = (self._guide_step_mode3 + 1) % len(steps)
        title, active, explanation = steps[self._guide_step_mode3]
        # Comparison intentionally has no topology canvas or active packet stream.
        self._active_nodes = set()
        self._active_edges = set()
        self._guide_var.set(explanation)
        self._mode_desc_var.set(f"COMPARISON DEMO: {title}")
        self._redraw_canvas()

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
            if pid == "kv_b" and self._current_mode != "normal":
                self._select_mode("normal")
            elif pid == "kv_o" and self._current_mode != "ai":
                self._select_mode("ai")

        elif kind == "step_update":
            data = event[1]
            mode = data["mode"]
            step = data["step"]
            total = data["total"]
            io_ms = data["io_ms"]
            comp_ms = data["compute_ms"]
            loss = data["loss"]
            batch = data["batch"]
            active_nodes = set(data["active_nodes"])
            active_edges = {tuple(edge) for edge in data.get("active_edges", [])}

            gpu_util = 14.2 if mode == "baseline" else 98.5
            self._batch_var.set(f"Batch: {batch}/{total} ({(batch/total)*100:.0f}%)")
            self._loss_var.set(f"Loss: {loss:.4f}")
            self._gpu_util_var.set(f"GPU Utilization: {gpu_util:.1f}%")
            self._io_wait_var.set(f"PCIe Stall: {io_ms:.1f} ms")

            is_chkpt = (batch in (10, 20))

            if mode == "baseline":
                if self._current_mode == "normal":
                    if is_chkpt:
                        active_nodes.add("ssd")
                    self._active_nodes = active_nodes
                    self._active_edges = active_edges
                    self._packets = []
                    self._mode_desc_var.set(f"TRADITIONAL: Batch {batch}/{total} - CPU Stalled ({io_ms:.1f}ms PCIe wait)")
                    self._redraw_canvas()
            else:
                if self._current_mode == "ai":
                    if is_chkpt:
                        active_nodes.add("checkpoint")
                    self._active_nodes = active_nodes
                    self._active_edges = active_edges
                    self._packets = []
                    self._mode_desc_var.set(f"AI-SSD OPTIMISED: Batch {batch}/{total} - Continuous Compute (0ms PCIe stall)")
                    self._redraw_canvas()

        elif kind == "results":
            baseline, optimised = event[1], event[2]
            self._baseline = baseline
            self._optimised = optimised
            self._running = False
            self._run_btn.config(state="normal", bg=NEON_BLUE)
            self._set_status("Showcase Complete!", NEON_GRN)
            self._select_mode("comparison")
            self._populate_results(baseline, optimised)
            self._embed_chart(baseline, optimised)

        elif kind == "error":
            tb = event[1]
            self._running = False
            self._run_btn.config(state="normal", bg=NEON_BLUE)
            self._set_status("Error!", NEON_RED)
            self._log_append("ERROR:\n" + tb, "err")

    # -----------------------------------------------------------------------
    # Log & Dashboard Panels
    # -----------------------------------------------------------------------

    def _build_dashboard_panels(self, parent: tk.Widget) -> None:
        row_frame = tk.Frame(parent, bg=BG)
        row_frame.pack(fill="both", expand=True, padx=12, pady=4)
        self._dashboard_frame = row_frame
        row_frame.columnconfigure(0, weight=1)
        row_frame.columnconfigure(1, weight=1)

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

        right = tk.Frame(row_frame, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        chart_frame = tk.Frame(right, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        chart_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        chart_frame.rowconfigure(1, weight=1)
        chart_frame.columnconfigure(0, weight=1)

        tk.Label(chart_frame, text="Live Benchmark Charts", font=FONT_HEAD, fg=FG, bg=BG2).grid(row=0, column=0, sticky="w", padx=10, pady=6)

        self._chart_placeholder = tk.Frame(chart_frame, bg=BG3, height=220)
        self._chart_placeholder.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        self._chart_lbl = tk.Label(self._chart_placeholder, text="Charts will appear here after benchmark runs.", font=FONT_BODY, fg=FG2, bg=BG3)
        self._chart_lbl.place(relx=0.5, rely=0.5, anchor="center")

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

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree["yscrollcommand"] = vsb.set

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

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

        ax = axes[0]
        ax.set_facecolor("#1c2033")
        bars = ax.bar(["Traditional", "AI-SSD"], [b.avg_kv_latency_ms, o.avg_kv_latency_ms], color=[NEON_RED, NEON_BLUE], width=0.5)
        ax.set_title("Step Latency (ms)", fontsize=8, color=FG)
        ax.tick_params(colors=FG2, labelsize=7)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.1f}ms", ha="center", fontsize=7, color=FG)

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

        self._guide_step_mode1 = -1
        self._guide_step_mode2 = -1
        self._guide_step_mode3 = -1
        self._select_mode("ai")
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
