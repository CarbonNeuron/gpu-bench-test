# GpuBench Design Document

**Date:** 2026-02-23
**Status:** Approved

## Goal

A CLI tool that benchmarks all available compute devices (CPU, CUDA, OpenCL) using ILGPU and displays results with Spectre.Console. Produces comparable numbers across devices for memory bandwidth, compute throughput, matrix multiplication, latency, and memory access patterns.

## Tech Stack

- .NET 9 (latest stable)
- ILGPU + ILGPU.Algorithms — cross-platform GPU compute
- Spectre.Console + Spectre.Cli — terminal UI and CLI parsing
- System.Text.Json — export

## Architecture

Single Spectre.Cli default command with flags: `--suite`, `--device`, `--export`, `--quick`, `--size`, `--list`, `--full`.

### CLI Interface

```
dotnet run                              # all suites, all devices
dotnet run -- --suite memory            # specific suite
dotnet run -- --device 0                # specific device by index
dotnet run -- --device cuda             # specific device by type
dotnet run -- --list                    # list devices only
dotnet run -- --export results.json     # export to JSON
dotnet run -- --export results.md       # export to Markdown
dotnet run -- --quick                   # smaller sizes, fewer iterations
dotnet run -- --suite matmul --size 8192
```

### Core Interface

```csharp
interface IBenchmarkSuite {
    string Name { get; }
    string Description { get; }
    bool SupportsDevice(Device device);
    Task<List<BenchmarkResult>> RunAsync(
        IReadOnlyList<Accelerator> accelerators,
        BenchmarkOptions options,
        ProgressTracker progress);
}
```

Each suite returns `BenchmarkResult` objects consumed by `ResultsRenderer` (display) and `ExportWriter` (JSON/Markdown).

## Project Structure

```
GpuBench/
  GpuBench.csproj
  Program.cs                    # Entry point, Spectre.Cli setup, orchestration
  Benchmarks/
    IBenchmarkSuite.cs          # Suite interface
    DeviceInfo.cs               # Device enumeration and info display
    MemoryBandwidth.cs          # H→D, D→H, D→D transfer benchmarks
    ComputeThroughput.cs        # FP32/FP64/Int32 throughput + vector ops
    MatrixMultiply.cs           # Naive + tiled matmul with scaling
    LatencyTests.cs             # Kernel launch, alloc, sync overhead
    MemoryPatterns.cs           # Sequential/random/coalesced access patterns
  Kernels/
    ComputeKernels.cs           # FMA kernels for all precisions
    MatMulKernels.cs            # Naive + tiled matmul kernels
    MemoryKernels.cs            # Memory pattern test kernels
  Rendering/
    ResultsRenderer.cs          # Spectre tables, bar charts, winner panels
    ProgressTracker.cs          # Live progress wrapper
    ExportWriter.cs             # JSON + Markdown export
  Models/
    BenchmarkResult.cs          # Result data model
    DeviceProfile.cs            # Device info model
    BenchmarkOptions.cs         # CLI options model
```

## Benchmark Suites

### 1. Device Info (always first)

Spectre table with: Name, Type (CPU/CUDA/OpenCL), Compute Units, Max Threads/Group, Max Shared Memory, Memory Size, Clock Rate, Warp Size. Color-coded panels per device.

### 2. Memory Bandwidth (`--suite memory`)

- **H→D Transfer:** Sizes 1MB–1GB, 5 runs each, bandwidth in GB/s
- **D→H Transfer:** Same sizes and methodology
- **D→D Copy:** Same sizes between two device buffers
- CPU excluded from this suite

### 3. Compute Throughput (`--suite compute`)

- **FP32 GFLOPS:** FMA-heavy kernel, 1M/4M/16M threads
- **FP64 GFLOPS:** Same with doubles
- **Int32 GOPS:** Same with int multiply-add
- **Vector Ops:** Element-wise add/mul/fma on 64M elements (realistic workload)

### 4. Matrix Multiplication (`--suite matmul`)

- **Naive kernel:** One thread per output element, sizes 1024–8192
- **Tiled kernel:** 16x16 shared memory tiles, same sizes
- **Scaling analysis:** 512–8192, GFLOPS vs size
- CPU limited to sizes <= 2048 unless `--full`
- Cross-device result verification (first device = reference, epsilon check)

### 5. Latency (`--suite latency`)

- **Kernel launch overhead:** Empty kernel, 100 launches, min/avg/max/p99
- **Memory allocation:** Time Allocate1D for various sizes, 50 iterations
- **Sync overhead:** Time Synchronize() idle vs after kernel

### 6. Memory Patterns (`--suite patterns`)

- **Sequential vs random read:** 64M float buffer, measure cache effectiveness
- **Shared vs global memory:** N repeated reads comparison
- **Coalesced vs strided:** Adjacent vs strided access, strides 2–32

## Cross-Device Verification

Applies to matmul and vector operations. First device = reference. Subsequent devices checked within epsilon (1e-2 FP32, 1e-6 FP64). Marked with checkmark or warning.

## Rendering

- **During runs:** `AnsiConsole.Progress()` with per-benchmark tasks, status spinners
- **Results:** Spectre tables with colored headers (cyan=AMD/OpenCL, green=NVIDIA/CUDA, yellow=CPU), bold best values, right-aligned numbers, thousands separators
- **Summaries:** Bar charts for visual comparison, winner panels after each suite
- **Overall:** Summary table at end, overall champion callout

## Safety

- Check `device.MemorySize` before allocating; skip sizes > 80% of device memory
- Per-device try/catch in each benchmark; failures logged, others continue
- All buffers/accelerators in `using` blocks
- 1-2 warmup iterations excluded from timing
- `Stopwatch` timing with `accelerator.Synchronize()` before stop

## Statistical Reporting

Minimum 3 iterations (5 standard, 10 for latency):
- Best, average, worst, standard deviation
- Comparison tables use best (peak capability)

## Export

- **JSON:** Timestamped, includes system info, device info, all results
- **Markdown:** Formatted tables for docs/GitHub issues
