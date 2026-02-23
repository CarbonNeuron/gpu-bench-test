# GpuBench

Cross-platform GPU and CPU benchmarking CLI built with .NET 9, ILGPU, and Spectre.Console.

## Features

- **Compute Throughput** -- FP32, FP64, Int32, and vector operation benchmarks (GFLOPS/GOPS)
- **Matrix Multiplication** -- Naive and tiled (shared memory) kernels with scaling analysis
- **Memory Bandwidth** -- Host-to-device, device-to-host, and device-to-device transfer rates
- **Latency Tests** -- Kernel launch overhead, memory allocation latency, synchronization cost
- **Memory Patterns** -- Sequential vs random, shared vs global, coalesced vs strided access
- **Multi-device** -- Benchmarks every GPU and CPU, with cross-device verification
- **Export** -- Results in JSON or Markdown format
- **Rich output** -- Tables, bar charts, and winner panels via Spectre.Console

## Quick Start

```bash
# Clone
git clone https://github.com/CarbonNeuron/gpu-bench-test.git
cd gpu-bench-test

# Run all benchmarks on all devices
dotnet run --project GpuBench

# Quick mode (fewer iterations, smaller sizes)
dotnet run --project GpuBench -- --quick
```

## Usage

```
gpubench [OPTIONS]

OPTIONS:
    --list              List available devices and exit
    --suite <SUITE>     Run a specific benchmark suite
    --device <DEVICE>   Target a specific device
    --export <PATH>     Export results to file (.json or .md)
    --quick             Quick mode: smaller sizes, fewer iterations
    --full              Full mode: include large sizes and CPU for all benchmarks
    --size <SIZE>       Custom matrix size for the matmul suite
```

### Examples

```bash
# List detected devices
dotnet run --project GpuBench -- --list

# Run only compute throughput benchmarks
dotnet run --project GpuBench -- --suite compute

# Benchmark a specific device
dotnet run --project GpuBench -- --device cuda

# Export results as JSON
dotnet run --project GpuBench -- --export results.json

# Export results as Markdown
dotnet run --project GpuBench -- --export results.md

# Full matrix multiply with custom size
dotnet run --project GpuBench -- --suite matmul --size 4096

# Quick benchmarks on GPU, exported to JSON
dotnet run --project GpuBench -- --quick --device cuda --export results.json
```

## Benchmark Suites

| Suite | Flag | What it measures |
|-------|------|-----------------|
| Compute Throughput | `--suite compute` | FP32/FP64 FMA, Int32 ops, vector add/FMA (GFLOPS) |
| Matrix Multiply | `--suite matmul` | Naive and 16x16 tiled matmul with scaling (GFLOPS) |
| Memory Bandwidth | `--suite memory` | H->D, D->H, D->D transfers from 1 MB to 1 GB (GB/s) |
| Latency | `--suite latency` | Kernel launch, buffer allocation, sync overhead (us) |
| Memory Patterns | `--suite patterns` | Sequential/random, shared/global, coalesced/strided access |

## Export Formats

**JSON** (`.json`) -- Machine-readable output with system info, device profiles, and all results including best/average/worst/stddev statistics and verification status.

**Markdown** (`.md`) -- Human-readable tables grouped by suite with per-device columns and a summary section highlighting winners.

## Building from Source

```bash
dotnet restore GpuBench/GpuBench.csproj
dotnet build GpuBench/GpuBench.csproj -c Release
```

To publish a self-contained single-file binary:

```bash
# Linux
dotnet publish GpuBench/GpuBench.csproj -c Release -r linux-x64 --self-contained -p:PublishSingleFile=true -o ./publish

# Windows
dotnet publish GpuBench/GpuBench.csproj -c Release -r win-x64 --self-contained -p:PublishSingleFile=true -o ./publish
```

## Download

Pre-built binaries for Linux and Windows are available on the [Releases](https://github.com/CarbonNeuron/gpu-bench-test/releases) page.

## Requirements

- .NET 9 SDK (for building from source)
- GPU drivers: NVIDIA CUDA or OpenCL-compatible drivers for GPU benchmarks
- CPU benchmarks work on any platform without additional drivers
