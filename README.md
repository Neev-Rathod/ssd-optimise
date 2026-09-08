# AI-Era SSD Emulator

This project simulates and benchmarks two ways of serving model-checkpoint data:

- **Standard I/O:** reads the file with normal `open()` and 4 KB chunks.
- **AI-SSD emulation:** uses memory mapping, OS readahead hints, zero-copy views, and predictive prefetching.

It also simulates LLM-style KV-cache inference, comparing synchronous block loading with asynchronous prefetching.

> This is a software simulation, not a driver or a replacement for physical SSD benchmarking.

## AI-Era SSD proposal

The optimised path models an SSD designed to keep AI accelerators supplied with data while avoiding unnecessary copies:

- **Tensor-aware firmware QoS:** place sequential training shards together, protect low-tail-latency read queues, and schedule checkpoint writes out of the critical path.
- **Adaptive DRAM/SLC caching:** retain hot KV-cache blocks and model metadata, with confidence-gated prefetching so speculative reads do not waste NAND endurance or energy.
- **Computational storage engine:** use controller cores for decompression, checksums, and simple tensor filtering before DMA, reducing transferred bytes and accelerator wakeups.
- **Direct, observable data plane:** stream SSD data toward VRAM with GPUDirect today and CXL-ready interfaces later; expose queue depth, cache confidence, and energy telemetry to the runtime.

These ideas lower tail latency and PCIe traffic while retaining a clear host-visible data path and predictable QoS.

## Requirements

- Windows
- Python 3.10 or newer
- Internet access during setup to install dependencies

## Setup

From the project directory, run:

```bat
setup.bat
```

The script creates the `ai_ssd_env` virtual environment and installs NumPy, psutil, Matplotlib, and CPU-only PyTorch.

Activate the environment when needed:

```bat
ai_ssd_env\Scripts\activate.bat
```

## Run the Benchmark

Command-line benchmark:

```bat
python emulator.py
```

The benchmark creates a 100 MB model-checkpoint-like file, then compares:

1. Dataset loading latency and throughput.
2. RAM usage during loading.
3. KV-cache inference latency per step.
4. KV-cache hit rate and peak RAM usage.

## Run the GUI

```bat
python gui.py
```

The GUI includes:

- Live benchmark phase cards.
- Separate **Normal / Standard I/O** and **AI-SSD Optimised I/O** hardware paths.
- Activity indicators for the application, CPU, RAM, OS page cache, SSD, and prefetch engine.
- A `Next Step` button for walking through the data flow manually.
- A scrollable dashboard with live logs, charts, and a results table.

## Project Outputs

| Path | Purpose |
| --- | --- |
| `data/model_checkpoint.bin` | Generated benchmark dataset |
| `results/benchmark_results.png` | Comparison chart |
| `logs/emulator.log` | Runtime log |

The generated dataset, logs, charts, virtual environment, and Python cache files are local runtime artifacts. Python cache files are excluded by `.gitignore`.

## Project Files

- `emulator.py` - benchmark engine, simulated readers, KV-cache models, and chart generation.
- `gui.py` - Tkinter dashboard for running and explaining the benchmark.
- `setup.bat` - Windows environment setup script.
