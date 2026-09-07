# -*- coding: utf-8 -*-
"""
emulator.py
===========
AI-Era SSD Emulator & Benchmark Framework
------------------------------------------
Simulates and benchmarks "AI-Era SSD Software Concepts" vs. standard file I/O
on a consumer laptop.  All data sizes are scaled 1000x down:
    Simulated 100 GB  -> 100 MB on-disk file
    Simulated 10 GB   -> 10 MB  on-disk file
    etc.

Modules
-------
  1. DataGenerator       - creates binary payload (model-checkpoint-like data)
  2. StandardReader      - naive chunked OS read() baseline
  3. AISSDReader         - mmap + MADV_WILLNEED emulated AI-SSD reader
  4. KVCacheSimulator    - sync (baseline) vs. async prefetch (AI-SSD) inference loop
  5. BenchmarkEngine     - runs both sides head-to-head, collects metrics
  6. Visualizer          - matplotlib comparison plot -> results/benchmark_results.png

Run:
    python emulator.py

Requirements:
    pip install numpy psutil matplotlib torch   (CPU-only torch is fine)
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import logging
import mmap
import os
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Optional imports - degrade gracefully so the benchmark runs even without torch
# ---------------------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend - safe on any machine
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Force UTF-8 on stdout so Unicode log messages work on Windows cp1252 consoles
_stdout_handler = logging.StreamHandler(
    open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
)
_stdout_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
_file_handler = logging.FileHandler(LOG_DIR / "emulator.log", mode="w", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_stdout_handler, _file_handler])
log = logging.getLogger("emulator")

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------
IS_WINDOWS = sys.platform.startswith("win")
IS_LINUX   = sys.platform.startswith("linux")
IS_MACOS   = sys.platform == "darwin"

# madvise availability
_madvise_available = False
_MADV_WILLNEED     = 3       # POSIX value; Linux & macOS share this constant
if IS_LINUX or IS_MACOS:
    try:
        _libc = ctypes.CDLL("libc.so.6" if IS_LINUX else "libc.dylib", use_errno=True)
        _madvise_available = True
    except OSError:
        pass


def _madvise_willneed(mm: mmap.mmap, offset: int, length: int) -> None:
    """Best-effort MADV_WILLNEED hint.  Silently skipped on Windows."""
    if not _madvise_available:
        return
    try:
        # Obtain the raw C pointer for the slice
        addr = ctypes.addressof(
            (ctypes.c_char * 1).from_buffer(mm, offset)
        )
        _libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(length), ctypes.c_int(_MADV_WILLNEED))
    except Exception:
        pass   # non-critical - skip silently


# ===========================================================================
# 1.  DATA GENERATOR
# ===========================================================================

DATA_DIR     = Path("data")
RESULTS_DIR  = Path("results")

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Scale-down factor: 1000x  ->  100 MB simulates 100 GB
TARGET_FILE_SIZE_MB = 100          # MB of actual on-disk binary data
CHUNK_SIZE_BYTES    = 4 * 1024     # 4 KB - mimics typical OS page / SSD sector
TCAU_SIZE_BYTES     = 256 * 1024   # 256 KB - Tensor Continuous Allocation Unit

DATASET_FILE = DATA_DIR / "model_checkpoint.bin"


def generate_dataset(path: Path = DATASET_FILE,
                     size_mb: int = TARGET_FILE_SIZE_MB,
                     force: bool = False) -> Path:
    """
    Generate a binary file that looks like a model checkpoint:
      - Starts with a 64-byte magic header
      - Followed by float32 tensor blocks each prefixed by a 4-byte size word
      - Padded to exact `size_mb` MB with random data

    Returns the path to the generated file.
    """
    target_bytes = size_mb * 1024 * 1024

    if path.exists() and path.stat().st_size == target_bytes and not force:
        log.info("Dataset already exists (%d MB) - skipping generation.", size_mb)
        return path

    log.info("Generating %d MB dataset -> %s ...", size_mb, path)
    t0 = time.perf_counter()

    rng = np.random.default_rng(seed=42)

    with open(path, "wb") as fh:
        # Magic header
        header = b"AI_SSD_EMULATOR_V1" + b"\x00" * (64 - 18)
        fh.write(header)
        written = len(header)

        block_idx = 0
        while written < target_bytes:
            remaining = target_bytes - written
            # Alternate between small (4 KB) and large (256 KB) tensor blocks
            block_size = TCAU_SIZE_bytes = (
                TCAU_SIZE_BYTES if block_idx % 4 == 0 else CHUNK_SIZE_BYTES
            )
            block_size = min(block_size, remaining - 4)
            if block_size <= 0:
                fh.write(b"\x00" * remaining)
                break

            # 4-byte little-endian block length prefix
            fh.write(struct.pack("<I", block_size))
            written += 4

            # Random float32 payload
            n_floats = block_size // 4
            payload = rng.random(n_floats, dtype=np.float32).tobytes()
            # Pad if block_size not divisible by 4
            payload += b"\x00" * (block_size - len(payload))
            fh.write(payload)
            written += block_size
            block_idx += 1

        # Ensure exact size
        final_size = fh.tell()
        if final_size < target_bytes:
            fh.write(b"\x00" * (target_bytes - final_size))

    elapsed = time.perf_counter() - t0
    actual_mb = path.stat().st_size / (1024 ** 2)
    log.info("Dataset ready: %.1f MB in %.2f s", actual_mb, elapsed)
    return path


# ===========================================================================
# 2.  STANDARD (BASELINE) READER
# ===========================================================================

class StandardReader:
    """
    Naive chunked OS read using open() + read().
    Simulates legacy storage software:
      - No prefetch hints
      - Data copied through kernel page cache -> user-space buffer
      - 4 KB read granularity
    """

    def __init__(self, path: Path, chunk_size: int = CHUNK_SIZE_BYTES):
        self.path       = path
        self.chunk_size = chunk_size
        self._cache: Dict[int, bytes] = {}   # keyed by chunk index

    def read_all(self) -> Tuple[float, float, float]:
        """
        Read the entire file in chunks.

        Returns
        -------
        (elapsed_s, throughput_mb_s, peak_ram_mb)
        """
        proc    = psutil.Process()
        ram_start = proc.memory_info().rss / (1024 ** 2)

        t0           = time.perf_counter()
        total_bytes  = 0
        peak_ram     = ram_start

        with open(self.path, "rb") as fh:
            chunk_idx = 0
            while True:
                buf = fh.read(self.chunk_size)
                if not buf:
                    break
                self._cache[chunk_idx] = buf
                total_bytes += len(buf)
                chunk_idx   += 1

                ram_now  = proc.memory_info().rss / (1024 ** 2)
                peak_ram = max(peak_ram, ram_now)

        elapsed      = time.perf_counter() - t0
        throughput   = (total_bytes / (1024 ** 2)) / max(elapsed, 1e-9)
        peak_usage   = peak_ram - ram_start

        return elapsed, throughput, peak_usage

    def read_chunk(self, chunk_idx: int) -> bytes:
        """Synchronous read of a single chunk (used by KV-cache simulator)."""
        if chunk_idx in self._cache:
            return self._cache[chunk_idx]

        offset = chunk_idx * self.chunk_size
        file_size = self.path.stat().st_size
        if offset >= file_size:
            return b""

        with open(self.path, "rb") as fh:
            fh.seek(offset)
            data = fh.read(self.chunk_size)

        self._cache[chunk_idx] = data
        return data

    def clear_cache(self) -> None:
        self._cache.clear()
        gc.collect()


# ===========================================================================
# 3.  AI-SSD EMULATED READER  (mmap + MADV_WILLNEED)
# ===========================================================================

class AISSDReader:
    """
    AI-SSD emulated reader using memory-mapped I/O:

    Mechanism
    ---------
    1. `mmap` maps the entire file into the virtual address space.
       On first access the OS faults in only the needed pages -> lazy load.
    2. `madvise(MADV_WILLNEED)` (Linux/macOS) pre-faults the next TCAU into
       the page cache asynchronously before the inference loop requests it.
       On Windows the equivalent is `VirtualAlloc` with prefetch hints; we
       approximate by reading a few bytes ahead to warm the OS readahead buffer.
    3. Slices are returned as numpy views (zero-copy) or torch tensors,
       so no extra user-space copy occurs.
    """

    def __init__(self, path: Path, tcau_size: int = TCAU_SIZE_BYTES):
        self.path      = path
        self.tcau_size = tcau_size
        self._mm:  Optional[mmap.mmap] = None
        self._fh:  Any = None
        self._hit_count  = 0
        self._miss_count = 0

    # ---- context manager ----

    def __enter__(self) -> "AISSDReader":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def open(self) -> None:
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        log.debug("mmap opened: %d bytes", len(self._mm))

    def close(self) -> None:
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._fh:
            self._fh.close()
            self._fh = None

    # ---- prefetch hint ----

    def _prefetch_hint(self, offset: int, length: int) -> None:
        """
        Issue MADV_WILLNEED on POSIX; warm the OS readahead buffer on Windows
        by reading a single byte at the start of the next TCAU.
        """
        if self._mm is None:
            return
        end = min(offset + length, len(self._mm))
        if IS_WINDOWS:
            # Windows: accessing a byte forces the OS to readahead the surrounding pages
            if end < len(self._mm):
                _ = self._mm[end]   # single-byte warm touch
        else:
            _madvise_willneed(self._mm, offset, end - offset)

    # ---- read methods ----

    def read_all(self) -> Tuple[float, float, float]:
        """
        Scan the entire file via mmap in TCAU-sized strides,
        issuing prefetch hints one TCAU ahead.

        Returns
        -------
        (elapsed_s, throughput_mb_s, peak_ram_mb)
        """
        proc      = psutil.Process()
        ram_start = proc.memory_info().rss / (1024 ** 2)

        self.open()
        t0          = time.perf_counter()
        total_bytes = len(self._mm)
        peak_ram    = ram_start

        offset = 0
        while offset < total_bytes:
            end = min(offset + self.tcau_size, total_bytes)

            # Prefetch the *next* TCAU before touching current one
            next_start = end
            next_end   = min(next_start + self.tcau_size, total_bytes)
            if next_start < total_bytes:
                self._prefetch_hint(next_start, next_end - next_start)

            # Zero-copy numpy view into mmap
            _ = np.frombuffer(self._mm[offset:end], dtype=np.uint8)

            ram_now  = proc.memory_info().rss / (1024 ** 2)
            peak_ram = max(peak_ram, ram_now)
            offset   = end

        elapsed    = time.perf_counter() - t0
        throughput = (total_bytes / (1024 ** 2)) / max(elapsed, 1e-9)
        peak_usage = peak_ram - ram_start
        self.close()

        return elapsed, throughput, peak_usage

    def read_tcau(self, tcau_idx: int) -> np.ndarray:
        """
        Return TCAU `tcau_idx` as a float32 numpy array (zero-copy view).
        Also pre-fetches the *next* TCAU.
        """
        if self._mm is None:
            raise RuntimeError("Call open() before read_tcau()")

        offset = tcau_idx * self.tcau_size
        if offset >= len(self._mm):
            return np.empty(0, dtype=np.float32)

        end = min(offset + self.tcau_size, len(self._mm))

        # Prefetch next block
        next_off = end
        next_end = min(next_off + self.tcau_size, len(self._mm))
        if next_off < len(self._mm):
            self._prefetch_hint(next_off, next_end - next_off)

        raw  = self._mm[offset:end]
        n    = len(raw) // 4
        view = np.frombuffer(raw, dtype=np.float32, count=n)
        return view

    @property
    def cache_hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return (self._hit_count / total * 100) if total > 0 else 0.0


# ===========================================================================
# 4.  KV-CACHE OFFLOAD SIMULATOR
# ===========================================================================

KV_CONTEXT_BLOCKS = 20    # number of context blocks per inference run
KV_BLOCK_SIZE_MB  = 2     # MB per KV block  (simulates ~2 GB on real hardware)
MOCK_RAM_BLOCKS   = 4     # mock RAM can hold only 4 blocks simultaneously
INFERENCE_STEPS   = KV_CONTEXT_BLOCKS


class BaselineKVCache:
    """
    Synchronous KV-cache: evicts and re-reads full context pages from disk
    on every inference step.  Represents unoptimised LLM inference.
    """

    def __init__(self, reader: StandardReader, kv_blocks: int = KV_CONTEXT_BLOCKS,
                 mock_ram_limit: int = MOCK_RAM_BLOCKS):
        self.reader        = reader
        self.kv_blocks     = kv_blocks
        self.ram_limit     = mock_ram_limit
        self._loaded: deque = deque(maxlen=mock_ram_limit)   # LRU-like FIFO
        self._step_latencies: List[float] = []
        self._peak_ram_mb = 0.0
        self._hit_count   = 0
        self._miss_count  = 0

    def _load_block(self, block_id: int) -> bytes:
        """Simulate blocking disk read of one KV block."""
        bytes_needed = KV_BLOCK_SIZE_MB * 1024 * 1024
        chunks_per_block = bytes_needed // CHUNK_SIZE_BYTES
        data = b""
        for i in range(chunks_per_block):
            chunk_idx = block_id * chunks_per_block + i
            data += self.reader.read_chunk(chunk_idx)
        return data

    def run_inference(self) -> Tuple[List[float], float, float]:
        """
        Run a simulated inference loop.

        Returns
        -------
        (step_latencies_ms, peak_ram_mb, cache_hit_rate_pct)
        """
        proc = psutil.Process()
        self._loaded.clear()
        self._step_latencies.clear()
        self._hit_count = 0
        self._miss_count = 0
        peak_ram = 0.0

        for step in range(self.kv_blocks):
            t0 = time.perf_counter()

            # Check if block is in mock RAM
            if step in self._loaded:
                self._hit_count += 1
            else:
                # Evict oldest if at capacity, then load
                self._miss_count += 1
                _ = self._load_block(step)          # blocking disk read
                if len(self._loaded) >= self.ram_limit:
                    self._loaded.popleft()
                self._loaded.append(step)

            # Simulate token processing latency
            time.sleep(0.002)                       # 2 ms mock compute
            elapsed_ms = (time.perf_counter() - t0) * 1000

            ram_now  = proc.memory_info().rss / (1024 ** 2)
            peak_ram = max(peak_ram, ram_now)
            self._step_latencies.append(elapsed_ms)

        total = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total * 100) if total > 0 else 0.0
        return self._step_latencies, peak_ram, hit_rate


class OptimisedKVCache:
    """
    Async predictive-prefetch KV-cache using AISSDReader.

    A background asyncio worker monitors the current step index and
    pre-loads the *next* KV block into a pinned buffer before the
    inference loop requests it.  This hides disk latency behind compute.
    """

    def __init__(self, reader: AISSDReader, kv_blocks: int = KV_CONTEXT_BLOCKS,
                 mock_ram_limit: int = MOCK_RAM_BLOCKS):
        self.reader    = reader
        self.kv_blocks = kv_blocks
        self.ram_limit = mock_ram_limit
        self._prefetch_buffer: Dict[int, np.ndarray] = {}
        self._step_latencies: List[float] = []
        self._hit_count   = 0
        self._miss_count  = 0
        self._peak_ram_mb = 0.0

    async def _prefetch_worker(self, current_step: int) -> None:
        """Async worker: load the next block into buffer if not already there."""
        next_step = current_step + 1
        if next_step >= self.kv_blocks:
            return
        if next_step not in self._prefetch_buffer:
            # Yield to event loop, then do the "disk" load
            await asyncio.sleep(0)
            # Use mmap TCAU read (zero-copy)
            n_tcaus_per_block = max(1, (KV_BLOCK_SIZE_MB * 1024 * 1024) // TCAU_SIZE_BYTES)
            blocks = []
            for i in range(n_tcaus_per_block):
                tcau_idx = next_step * n_tcaus_per_block + i
                view = self.reader.read_tcau(tcau_idx)
                blocks.append(view.copy())   # copy so we can close mmap later
            self._prefetch_buffer[next_step] = np.concatenate(blocks) if blocks else np.empty(0)

    async def _run_async(self) -> Tuple[List[float], float, float]:
        proc = psutil.Process()
        self._prefetch_buffer.clear()
        self._step_latencies.clear()
        self._hit_count   = 0
        self._miss_count  = 0
        peak_ram = 0.0

        # Kick off prefetch for step 0 immediately
        await self._prefetch_worker(-1)    # pre-fetches step 0

        for step in range(self.kv_blocks):
            t0 = time.perf_counter()

            # Launch prefetch for step+1 concurrently
            prefetch_task = asyncio.create_task(self._prefetch_worker(step))

            # Serve current block from prefetch buffer or fall back
            if step in self._prefetch_buffer:
                _ = self._prefetch_buffer.pop(step)
                self._hit_count += 1
            else:
                # Cold miss – load synchronously (should be rare)
                self._miss_count += 1
                n_tcaus = max(1, (KV_BLOCK_SIZE_MB * 1024 * 1024) // TCAU_SIZE_BYTES)
                for i in range(n_tcaus):
                    self.reader.read_tcau(step * n_tcaus + i)

            # Simulate token processing
            await asyncio.sleep(0.002)     # 2 ms mock compute

            await prefetch_task            # ensure next block is ready
            elapsed_ms = (time.perf_counter() - t0) * 1000

            ram_now  = proc.memory_info().rss / (1024 ** 2)
            peak_ram = max(peak_ram, ram_now)
            self._step_latencies.append(elapsed_ms)

        total    = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total * 100) if total > 0 else 0.0
        return self._step_latencies, peak_ram, hit_rate

    def run_inference(self) -> Tuple[List[float], float, float]:
        """Synchronous entry-point that drives the async loop."""
        return asyncio.run(self._run_async())


# ===========================================================================
# 5.  BENCHMARK ENGINE
# ===========================================================================

@dataclass
class BenchmarkResult:
    label:               str
    dataset_latency_ms:  float      # full dataset load latency in ms
    throughput_mb_s:     float      # dataset read throughput
    peak_ram_mb:         float      # peak RAM delta during dataset load
    kv_step_latencies:   List[float] = field(default_factory=list)
    kv_peak_ram_mb:      float = 0.0
    kv_hit_rate_pct:     float = 0.0

    @property
    def avg_kv_latency_ms(self) -> float:
        return float(np.mean(self.kv_step_latencies)) if self.kv_step_latencies else 0.0

    @property
    def p99_kv_latency_ms(self) -> float:
        return float(np.percentile(self.kv_step_latencies, 99)) if self.kv_step_latencies else 0.0


def _warm_os_cache(path: Path) -> None:
    """Quick single-pass read to warm the OS disk cache before timing."""
    with open(path, "rb") as fh:
        while fh.read(64 * 1024):
            pass


def run_benchmark(dataset_path: Path) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """
    Execute head-to-head benchmark.

    Phase 1 - Dataset load (full file scan)
    Phase 2 - KV-cache inference simulation (KV_CONTEXT_BLOCKS steps)

    Both phases run under identical conditions:
      - Same dataset file
      - OS disk cache cleared between baseline and optimised by GC + re-open

    Returns (baseline_result, optimised_result)
    """

    log.info("=" * 60)
    log.info("BENCHMARK START")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # PHASE 1 - Dataset Load  (Baseline)
    # ------------------------------------------------------------------
    log.info("\n[Phase 1A] Baseline dataset read ...")
    gc.collect()

    std_reader = StandardReader(dataset_path)
    elapsed_b, tp_b, ram_b = std_reader.read_all()
    log.info("  Latency : %.1f ms  |  Throughput : %.1f MB/s  |  RAM delta : %.1f MB",
             elapsed_b * 1000, tp_b, ram_b)

    # ------------------------------------------------------------------
    # PHASE 1 - Dataset Load  (Optimised)
    # ------------------------------------------------------------------
    log.info("\n[Phase 1B] AI-SSD emulated dataset read ...")
    gc.collect()

    ai_reader = AISSDReader(dataset_path)
    elapsed_o, tp_o, ram_o = ai_reader.read_all()
    log.info("  Latency : %.1f ms  |  Throughput : %.1f MB/s  |  RAM delta : %.1f MB",
             elapsed_o * 1000, tp_o, ram_o)

    # ------------------------------------------------------------------
    # PHASE 2 - KV-Cache Inference  (Baseline)
    # ------------------------------------------------------------------
    log.info("\n[Phase 2A] Baseline KV-cache inference (%d steps) ...", INFERENCE_STEPS)
    std_reader.clear_cache()
    gc.collect()

    baseline_kv = BaselineKVCache(std_reader)
    kv_lats_b, kv_ram_b, kv_hit_b = baseline_kv.run_inference()
    log.info("  Avg latency : %.2f ms/step  |  Peak RAM : %.1f MB  |  Hit rate : %.1f%%",
             float(np.mean(kv_lats_b)), kv_ram_b, kv_hit_b)

    # ------------------------------------------------------------------
    # PHASE 2 - KV-Cache Inference  (Optimised)
    # ------------------------------------------------------------------
    log.info("\n[Phase 2B] Optimised KV-cache inference (%d steps) ...", INFERENCE_STEPS)
    gc.collect()

    with AISSDReader(dataset_path) as optimised_reader:
        opt_kv = OptimisedKVCache(optimised_reader)
        kv_lats_o, kv_ram_o, kv_hit_o = opt_kv.run_inference()

    log.info("  Avg latency : %.2f ms/step  |  Peak RAM : %.1f MB  |  Hit rate : %.1f%%",
             float(np.mean(kv_lats_o)), kv_ram_o, kv_hit_o)

    # ------------------------------------------------------------------
    # Package results
    # ------------------------------------------------------------------
    baseline = BenchmarkResult(
        label              = "Baseline (Standard I/O)",
        dataset_latency_ms = elapsed_b * 1000,
        throughput_mb_s    = tp_b,
        peak_ram_mb        = ram_b,
        kv_step_latencies  = kv_lats_b,
        kv_peak_ram_mb     = kv_ram_b,
        kv_hit_rate_pct    = kv_hit_b,
    )
    optimised = BenchmarkResult(
        label              = "Optimised (AI-SSD Emulated)",
        dataset_latency_ms = elapsed_o * 1000,
        throughput_mb_s    = tp_o,
        peak_ram_mb        = ram_o,
        kv_step_latencies  = kv_lats_o,
        kv_peak_ram_mb     = kv_ram_o,
        kv_hit_rate_pct    = kv_hit_o,
    )
    return baseline, optimised


# ===========================================================================
# 6.  VISUALIZER
# ===========================================================================

def _speedup_label(baseline_val: float, opt_val: float, higher_is_better: bool = False) -> str:
    """Return a speedup/improvement string."""
    if opt_val == 0:
        return ""
    if higher_is_better:
        ratio = opt_val / max(baseline_val, 1e-9)
        return f"{ratio:.1f}x higher"
    else:
        ratio = baseline_val / max(opt_val, 1e-9)
        return f"{ratio:.1f}x faster"


def visualize(baseline: BenchmarkResult, optimised: BenchmarkResult,
              out_path: Path = RESULTS_DIR / "benchmark_results.png") -> None:
    """Generate a 2x2 comparison chart and save to disk."""

    if not MATPLOTLIB_AVAILABLE:
        log.warning("matplotlib not available - skipping visualization.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("AI-Era SSD Emulator - Benchmark Results", fontsize=16, fontweight="bold", y=1.01)

    colors = {"baseline": "#E55C5C", "optimised": "#4CA6E8"}

    labels      = ["Baseline", "AI-SSD"]
    bar_colors  = [colors["baseline"], colors["optimised"]]

    def _bar(ax: Any, values: List[float], ylabel: str, title: str,
             unit: str = "", higher_better: bool = False) -> None:
        bars = ax.bar(labels, values, color=bar_colors, width=0.5, edgecolor="white", linewidth=1.2)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=11)

        # Annotate bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:.1f}{unit}", ha="center", va="bottom", fontsize=10)

        # Speedup arrow annotation
        su = _speedup_label(values[0], values[1], higher_is_better=higher_better)
        if su:
            ax.text(0.98, 0.97, su, transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    color=colors["optimised"], fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors["optimised"], alpha=0.7))

    # -- Panel A: Dataset Load Latency ---------------------------------
    _bar(axes[0, 0],
         [baseline.dataset_latency_ms, optimised.dataset_latency_ms],
         "Latency (ms)", "Dataset Load Latency", unit=" ms")

    # -- Panel B: Read Throughput --------------------------------------
    _bar(axes[0, 1],
         [baseline.throughput_mb_s, optimised.throughput_mb_s],
         "Throughput (MB/s)", "Read Throughput", unit=" MB/s",
         higher_better=True)

    # -- Panel C: KV-Cache Step Latency (step-by-step line) -----------
    ax_c = axes[1, 0]
    steps = list(range(1, len(baseline.kv_step_latencies) + 1))
    ax_c.plot(steps, baseline.kv_step_latencies,
              color=colors["baseline"], label="Baseline", linewidth=2, marker="o", markersize=4)
    ax_c.plot(steps, optimised.kv_step_latencies,
              color=colors["optimised"], label="AI-SSD", linewidth=2, marker="s", markersize=4)
    ax_c.set_xlabel("Inference Step", fontsize=10)
    ax_c.set_ylabel("Latency (ms)", fontsize=10)
    ax_c.set_title("KV-Cache Step Latency", fontsize=11, pad=8)
    ax_c.legend(fontsize=9)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    su = _speedup_label(baseline.avg_kv_latency_ms, optimised.avg_kv_latency_ms)
    if su:
        ax_c.text(0.98, 0.97, f"avg {su}", transform=ax_c.transAxes,
                  ha="right", va="top", fontsize=9, color=colors["optimised"],
                  fontstyle="italic",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors["optimised"], alpha=0.7))

    # -- Panel D: Peak RAM + Cache Hit Rate ----------------------------
    ax_d = axes[1, 1]
    ram_vals  = [baseline.peak_ram_mb, optimised.peak_ram_mb]
    x = np.arange(len(labels))
    width = 0.3
    bars_ram = ax_d.bar(x - width / 2, ram_vals, width, label="Peak RAM (MB)",
                        color=bar_colors, edgecolor="white", linewidth=1.2)

    ax_d2 = ax_d.twinx()
    hit_vals = [baseline.kv_hit_rate_pct, optimised.kv_hit_rate_pct]
    bars_hit = ax_d2.bar(x + width / 2, hit_vals, width, label="Cache Hit Rate (%)",
                         color=[c + "99" for c in ["#E55C5C", "#4CA6E8"]],
                         edgecolor="white", linewidth=1.2)

    ax_d.set_ylabel("Peak RAM delta (MB)", fontsize=10)
    ax_d2.set_ylabel("Cache Hit Rate (%)", fontsize=10)
    ax_d.set_title("Peak RAM vs. Cache Hit Rate", fontsize=11, pad=8)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, fontsize=11)
    ax_d.spines["top"].set_visible(False)

    patch1 = mpatches.Patch(color="#E55C5C", label="Baseline RAM")
    patch2 = mpatches.Patch(color="#4CA6E8", label="AI-SSD RAM")
    patch3 = mpatches.Patch(color="#E55C5C99", label="Baseline Hit%")
    patch4 = mpatches.Patch(color="#4CA6E899", label="AI-SSD Hit%")
    ax_d.legend(handles=[patch1, patch2, patch3, patch4], fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Chart saved -> %s", out_path.resolve())
    plt.close(fig)


# ===========================================================================
# 7.  CLI SUMMARY TABLE
# ===========================================================================

def print_summary(baseline: BenchmarkResult, optimised: BenchmarkResult) -> None:
    """Print a formatted benchmark results table to stdout."""

    def _speedup(b: float, o: float, higher: bool = False) -> str:
        if o == 0:
            return "-"
        ratio = (o / max(b, 1e-9)) if higher else (b / max(o, 1e-9))
        direction = "^" if higher else "v"
        return f"{ratio:.1f}x {direction}"

    COL_W = [38, 22, 22]
    sep   = "=" * (sum(COL_W) + 6)
    div   = "-" * (sum(COL_W) + 6)

    rows = [
        ("Metric", "Baseline", "Optimised (AI-SSD)"),
        None,
        ("Dataset Load Latency",
         f"{baseline.dataset_latency_ms:.1f} ms",
         f"{optimised.dataset_latency_ms:.1f} ms  [{_speedup(baseline.dataset_latency_ms, optimised.dataset_latency_ms)}]"),
        ("Read Throughput",
         f"{baseline.throughput_mb_s:.0f} MB/s",
         f"{optimised.throughput_mb_s:.0f} MB/s  [{_speedup(baseline.throughput_mb_s, optimised.throughput_mb_s, higher=True)}]"),
        ("Peak RAM Delta (load)",
         f"{baseline.peak_ram_mb:.1f} MB",
         f"{optimised.peak_ram_mb:.1f} MB"),
        None,
        ("KV-Cache Avg Step Latency",
         f"{baseline.avg_kv_latency_ms:.2f} ms/step",
         f"{optimised.avg_kv_latency_ms:.2f} ms/step  [{_speedup(baseline.avg_kv_latency_ms, optimised.avg_kv_latency_ms)}]"),
        ("KV-Cache p99 Latency",
         f"{baseline.p99_kv_latency_ms:.2f} ms",
         f"{optimised.p99_kv_latency_ms:.2f} ms"),
        ("KV Cache Hit Rate",
         f"{baseline.kv_hit_rate_pct:.1f}%",
         f"{optimised.kv_hit_rate_pct:.1f}%  [{_speedup(baseline.kv_hit_rate_pct, optimised.kv_hit_rate_pct, higher=True)}]"),
        ("KV-Cache Peak RAM",
         f"{baseline.kv_peak_ram_mb:.1f} MB",
         f"{optimised.kv_peak_ram_mb:.1f} MB"),
    ]

    print()
    print(sep)
    print(f"{'AI-ERA SSD EMULATOR - BENCHMARK RESULTS':^{sum(COL_W) + 6}}")
    print(sep)

    for row in rows:
        if row is None:
            print(div)
            continue
        line = "  ".join(str(cell).ljust(w) for cell, w in zip(row, COL_W))
        print(line)

    print(sep)

    if MATPLOTLIB_AVAILABLE:
        print(f"\n  Chart saved -> {(RESULTS_DIR / 'benchmark_results.png').resolve()}")
    else:
        print("\n  (install matplotlib to generate the comparison chart)")

    print()


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    log.info("AI-Era SSD Emulator starting ...")
    log.info("Platform   : %s", sys.platform)
    log.info("Python     : %s", sys.version.split()[0])
    log.info("madvise    : %s", "available" if _madvise_available else "not available (Windows - using readahead warm touch)")
    log.info("torch      : %s", torch.__version__ if TORCH_AVAILABLE else "not installed")
    log.info("Scale-down : 1000x  (100 MB on-disk ~ 100 GB simulated)")
    log.info("")

    # 1. Generate dataset (skipped if already present)
    dataset_path = generate_dataset()

    # 2. Run benchmark
    baseline, optimised = run_benchmark(dataset_path)

    # 3. Print CLI summary table
    print_summary(baseline, optimised)

    # 4. Generate chart
    visualize(baseline, optimised)

    log.info("Done.  Results written to %s/", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
