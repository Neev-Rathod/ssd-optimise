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

The benchmark also produces a hardware-level estimate for each path. This estimate
models storage, CPU, memory, PCIe, VRAM, and accelerator timing; it is not a
measurement of a physical SSD or GPU.

## Run the GUI

```bat
python gui.py
```

The GUI includes:

- Live benchmark phase cards.
- Separate **Normal / Standard I/O** and **AI-SSD Optimised I/O** hardware paths.
- Activity indicators for the application, CPU, RAM, OS page cache, SSD, and prefetch engine.
- A `Next Step` button for walking through the data flow manually.
- A **Full Screen** topology view for presentations and architecture walkthroughs.
- Fullscreen `<`, `>`, and `Next Step` controls; the left and right arrow keys also advance the walkthrough, and `Esc` exits fullscreen.
- Progressive topology rendering: the initial canvas is empty, future components remain hidden, and components from earlier steps stay visible but dimmed.
- Animated current-step highlights and moving data packets along active links.
- A scrollable dashboard with live logs, charts, and a results table.

### GUI walkthrough workflow

1. Start the GUI with `python gui.py`.
2. Select **Mode 1** for the traditional pipeline or **Mode 2** for the AI-SSD pipeline.
3. Click **Next Step** to reveal the first part of the selected data path. The current components and links are highlighted.
4. Continue advancing to reveal later stages. Previously visited components remain on the canvas in a muted style, while unreached components and subsystem boundaries stay hidden.
5. Click **Full Screen** to focus on the topology. Use **Previous Step** and **Next Step** to navigate manually. The keyboard `Left` and `Right` arrow keys always advance to the next step; they are intentionally not a back button. Press `Esc` or **Exit** to return to the dashboard.
6. Run **Live Benchmark** to replace the manual walkthrough with live batch activity, packet animation, logs, charts, and the final comparison table.

### Mode 1: Traditional pipeline

The walkthrough follows the extra-copy path:

`SSD -> OS page cache -> dedicated RAM -> PCIe -> GPU VRAM -> accelerator`

This path represents blocking reads, page-cache/user-buffer copying, PCIe transfer
latency, and accelerator stalls while data is being loaded.

The traditional walkthrough is ordered as: application read request, SSD read into
the OS page cache, page-cache copy into host RAM, host-to-device PCIe transfer and
GPU compute, then a checkpoint write back to the SSD. GPU-side decode, resize, or
normalization can happen only after the batch reaches VRAM.

### Mode 2: AI-SSD pipeline

The walkthrough follows the optimised path:

`SSD/data engine -> lookahead + tensor preparation -> GPUDirect Storage -> GPU VRAM -> GPU preprocessing/compute`

The final stage adds asynchronous checkpoint offload back to the SSD. The model
represents mmap-based access, predictive prefetching, optional near-storage
decompression/validation/decode/resize/normalization, zero-copy views, direct DMA,
and overlapped compute. Image resizing and augmentation are described as proposed
pipeline stages; this emulator does not manipulate real image tensors.

### What the SSD data engine does

The SSD data engine is the simulated controller-side layer between storage and the
accelerator. Its job is to keep the next tensor data ready without making the CPU
copy every block:

- **Predictive prefetching:** looks ahead at the next expected pages or KV-cache
  blocks and requests them before the accelerator needs them.
- **Async DMA / GPUDirect path:** moves data directly toward GPU VRAM through the
  PCIe fabric, reducing host-buffer and page-cache copies.
- **Tensor-aware scheduling:** gives training reads priority and keeps checkpoint
  writes off the critical compute path.
- **Near-data processing:** represents optional decompression, checksum, decode,
  resize, normalization, and simple tensor filtering close to the storage
  controller, reducing transferred bytes.
- **Adaptive caching:** keeps frequently reused metadata and KV-cache blocks in
  DRAM/SLC-style cache space, while avoiding blind prefetches.

This project emulates those behaviors in software with memory mapping, readahead
hints, zero-copy views, and background prefetching. GPU-side preprocessing is a
separate stage that occurs after data reaches VRAM. It does not modify an SSD
firmware controller, resize real images, or provide real GPUDirect hardware access.

## Modeled improvement

The GUI comparison walkthrough uses the following representative hardware-model
values:

| Metric                    | Traditional path | AI-SSD path | Modeled improvement |
| ------------------------- | ---------------: | ----------: | ------------------: |
| KV/inference step latency |         197.5 ms |     13.2 ms |        14.9x faster |
| Read throughput           |         382 MB/s |  2,736 MB/s |         7.2x higher |

These values describe the emulator's modeled scenario and can differ from the
runtime numbers shown after a local benchmark. The actual result depends on the
machine, filesystem cache state, dataset size, and installed dependencies. The
important design effect is the same: reduce host-side copies and blocking I/O,
then overlap prefetch, data movement, compute, and checkpoint writes. GPU
utilization is intentionally omitted because this emulator does not measure it
reliably.

## Project Outputs

| Path                            | Purpose                     |
| ------------------------------- | --------------------------- |
| `data/model_checkpoint.bin`     | Generated benchmark dataset |
| `results/benchmark_results.png` | Comparison chart            |
| `logs/emulator.log`             | Runtime log                 |

The generated dataset, logs, charts, virtual environment, and Python cache files are local runtime artifacts. Python cache files are excluded by `.gitignore`.

## Project Files

- `emulator.py` - benchmark engine, simulated readers, KV-cache models, and chart generation.
- `gui.py` - Tkinter dashboard for running and explaining the benchmark.
- `setup.bat` - Windows environment setup script.
