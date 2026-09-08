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
- Fullscreen **Previous Step** and **Next Step** controls; the keyboard `Left` and `Right` arrow keys always advance the walkthrough, and `Esc` exits fullscreen.
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

The five fullscreen steps are:

1. Application read request.
2. SSD read into the OS page cache.
3. Page-cache copy into host RAM.
4. Host RAM to GPU VRAM transfer, followed by GPU preprocessing and compute.
5. Checkpoint write from the training path back to the SSD.

### Mode 2: AI-SSD pipeline

The walkthrough follows the optimised path:

`SSD/data engine -> lookahead + tensor preparation -> GPUDirect Storage -> GPU VRAM -> GPU preprocessing/compute`

The final stage adds asynchronous checkpoint offload back to the SSD. The model
represents mmap-based access, predictive prefetching, optional near-storage
decompression/validation/decode/resize/normalization, zero-copy views, direct DMA,
and overlapped compute. Image resizing and augmentation are described as proposed
pipeline stages; this emulator does not manipulate real image tensors.

The five fullscreen steps are:

1. Request the next batch and perform predictive lookahead.
2. Prepare tensor data near storage, such as decompression, validation, decode,
   resize, or normalization when supported by the deployment.
3. Transfer the prepared batch directly into GPU VRAM through the GPUDirect-style
   DMA path.
4. Run GPU-side preprocessing and model computation.
5. Offload checkpoint writes asynchronously while the next batch is prepared.

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

## Current benchmark results

The following scores were produced by running `python emulator.py` on the default
100 MB dataset with 20 KV-cache inference steps. Runtime values can change with
filesystem cache state and machine load.

| Runtime metric               | Traditional I/O | AI-SSD emulated |             Improvement |
| ---------------------------- | --------------: | --------------: | ----------------------: |
| Dataset load latency         |         46.7 ms |         33.6 ms |             1.4x faster |
| Read throughput              |      2,140 MB/s |      2,975 MB/s |             1.4x higher |
| Dataset-load RAM delta       |        110.6 MB |        100.5 MB |              9.1% lower |
| KV-cache average step        |         2.40 ms |         1.70 ms |             1.4x faster |
| KV-cache hit rate            |            0.0% |           95.0% |  Optimized cache active |
| KV-cache peak RAM            |        218.8 MB |        218.9 MB | Approximately unchanged |
| Total benchmark time         |         0.095 s |         0.068 s |     28.6% less I/O time |
| Estimated total with compute |         2.095 s |         2.012 s |              4.0% lower |

### Cycle-model component scores

These timings come from the emulator's cycle-accurate hardware model for the same
100 MB payload. They explain the stages shown in the GUI; they are not measurements
of a physical computational-storage SSD, NVMe controller, or GPU.

| Traditional component                      |                  Modeled time |
| ------------------------------------------ | ----------------------------: |
| NVMe SSD controller + flash read           |                     15.472 ms |
| OS page-cache allocation/double copy       |                      3.281 ms |
| Host CPU syscall and IRQ queue             |                      0.005 ms |
| DDR5 RAM access                            |                      1.638 ms |
| PCIe/host-to-device and VRAM transfer path | 7.490 ms link + 0.105 ms VRAM |
| GPU tensor compute                         |                  1,000.000 ms |

| AI-SSD component                                  | Modeled time |
| ------------------------------------------------- | -----------: |
| NVMe storage + predictive SLC cache               |     7.506 ms |
| SSD data engine: decompress/checksum/filter block |     0.001 ms |
| Data-engine to GPUDirect DMA path                 |     7.490 ms |
| Direct PCIe write into GPU VRAM                   |     7.490 ms |
| Direct GPU HBM3 VRAM access                       |     0.105 ms |
| GPU tensor compute overlapped with data movement  | 1,000.000 ms |

The traditional path additionally models these links: SSD-to-page-cache 15.420 ms,
page-cache-to-RAM 1.639 ms, RAM-to-PCIe 1.639 ms, PCIe-to-VRAM 7.490 ms, and
VRAM-to-GPU 0.031 ms. The AI-SSD path models SSD-to-data-engine 7.491 ms,
data-engine-to-PCIe 7.490 ms, PCIe-to-VRAM 7.490 ms, and VRAM-to-GPU 0.031 ms.

The important design effect is to reduce host-side copies and blocking I/O, then
overlap prefetch, tensor preparation, data movement, GPU work, and checkpoint
writes. GPU utilization percentages are intentionally omitted because this
emulator does not measure them reliably.

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
