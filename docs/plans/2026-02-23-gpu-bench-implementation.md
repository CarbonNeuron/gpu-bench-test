# GpuBench Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a comprehensive GPU/CPU benchmarking CLI that tests all available compute devices with memory, compute, matmul, latency, and memory pattern benchmarks, rendering results with Spectre.Console.

**Architecture:** Single Spectre.Cli default command dispatching to `IBenchmarkSuite` implementations. ILGPU provides cross-platform GPU compute. Results flow through a central renderer for display and export.

**Tech Stack:** .NET 9, ILGPU, ILGPU.Algorithms, Spectre.Console, System.Text.Json

---

### Task 1: Project Scaffolding

**Files:**
- Create: `GpuBench/GpuBench.csproj`
- Create: `GpuBench/Program.cs`
- Create: `GpuBench/.gitignore`

**Step 1: Create .gitignore**

```gitignore
bin/
obj/
*.user
*.suo
.vs/
*.DotSettings.user
results.json
results.md
```

**Step 2: Create project file**

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net9.0</TargetFramework>
    <RootNamespace>GpuBench</RootNamespace>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="ILGPU" Version="1.5.1" />
    <PackageReference Include="ILGPU.Algorithms" Version="1.5.1" />
    <PackageReference Include="Spectre.Console" Version="0.49.1" />
    <PackageReference Include="Spectre.Console.Cli" Version="0.49.1" />
  </ItemGroup>
</Project>
```

Note: Check NuGet for latest stable versions of each package at build time. The versions above are approximate — use whatever `dotnet add package <name>` installs.

**Step 3: Create minimal Program.cs**

```csharp
using Spectre.Console.Cli;

namespace GpuBench;

public static class Program
{
    public static int Main(string[] args)
    {
        var app = new CommandApp<BenchmarkCommand>();
        app.Configure(config =>
        {
            config.SetApplicationName("gpubench");
            config.SetApplicationVersion("1.0.0");
        });
        return app.Run(args);
    }
}
```

**Step 4: Create stub BenchmarkCommand**

Create `GpuBench/BenchmarkCommand.cs`:

```csharp
using Spectre.Console;
using Spectre.Console.Cli;
using System.ComponentModel;

namespace GpuBench;

public sealed class BenchmarkSettings : CommandSettings
{
    [CommandOption("--suite <SUITE>")]
    [Description("Benchmark suite to run: memory, compute, latency, matmul, patterns")]
    public string? Suite { get; set; }

    [CommandOption("--device <DEVICE>")]
    [Description("Device to benchmark: index number, name substring, 'cuda', 'opencl', or 'cpu'")]
    public string? Device { get; set; }

    [CommandOption("--export <PATH>")]
    [Description("Export results to file (.json or .md)")]
    public string? Export { get; set; }

    [CommandOption("--quick")]
    [Description("Quick mode: smaller sizes, fewer iterations")]
    public bool Quick { get; set; }

    [CommandOption("--full")]
    [Description("Full mode: include large sizes and CPU for all benchmarks")]
    public bool Full { get; set; }

    [CommandOption("--size <SIZE>")]
    [Description("Specific matrix size for matmul suite")]
    public int? Size { get; set; }

    [CommandOption("--list")]
    [Description("List available devices and exit")]
    public bool List { get; set; }
}

public sealed class BenchmarkCommand : Command<BenchmarkSettings>
{
    public override int Execute(CommandContext context, BenchmarkSettings settings)
    {
        AnsiConsole.MarkupLine("[bold]GpuBench v1.0[/]");
        AnsiConsole.MarkupLine("Settings parsed successfully.");
        return 0;
    }
}
```

**Step 5: Restore and build**

Run: `dotnet restore GpuBench/GpuBench.csproj && dotnet build GpuBench/GpuBench.csproj`
Expected: Build succeeded.

**Step 6: Verify CLI parsing works**

Run: `dotnet run --project GpuBench -- --list`
Run: `dotnet run --project GpuBench -- --suite compute --quick`
Expected: Both print "GpuBench v1.0" and "Settings parsed successfully."

**Step 7: Commit**

```bash
git add -A && git commit -m "feat: project scaffolding with Spectre.Cli"
```

---

### Task 2: Models

**Files:**
- Create: `GpuBench/Models/BenchmarkOptions.cs`
- Create: `GpuBench/Models/DeviceProfile.cs`
- Create: `GpuBench/Models/BenchmarkResult.cs`

**Step 1: Create BenchmarkOptions**

```csharp
namespace GpuBench.Models;

public sealed class BenchmarkOptions
{
    public string? Suite { get; set; }
    public string? DeviceFilter { get; set; }
    public string? ExportPath { get; set; }
    public bool Quick { get; set; }
    public bool Full { get; set; }
    public int? MatrixSize { get; set; }
    public bool ListOnly { get; set; }

    public int Iterations => Quick ? 3 : 5;
    public int LatencyIterations => Quick ? 50 : 100;
    public int WarmupIterations => Quick ? 1 : 2;

    public static BenchmarkOptions FromSettings(BenchmarkSettings settings) => new()
    {
        Suite = settings.Suite?.ToLowerInvariant(),
        DeviceFilter = settings.Device?.ToLowerInvariant(),
        ExportPath = settings.Export,
        Quick = settings.Quick,
        Full = settings.Full,
        MatrixSize = settings.Size,
        ListOnly = settings.List,
    };
}
```

**Step 2: Create DeviceProfile**

```csharp
using ILGPU;
using ILGPU.Runtime;

namespace GpuBench.Models;

public sealed class DeviceProfile
{
    public required string Name { get; init; }
    public required string DeviceType { get; init; } // "CPU", "CUDA", "OpenCL"
    public required int DeviceIndex { get; init; }
    public int ComputeUnits { get; init; }
    public int MaxThreadsPerGroup { get; init; }
    public long MaxSharedMemoryPerGroup { get; init; }
    public long MemorySize { get; init; }
    public int WarpSize { get; init; }
    public int ClockRate { get; init; }
    public string? DriverVersion { get; init; }

    public string Color => DeviceType switch
    {
        "CUDA" => "green",
        "OpenCL" => "cyan",
        "CPU" => "yellow",
        _ => "white"
    };

    public bool IsCpu => DeviceType == "CPU";

    public static DeviceProfile FromAccelerator(Accelerator accelerator, int index)
    {
        var device = accelerator.Device;
        return new DeviceProfile
        {
            Name = device.Name,
            DeviceType = device.AcceleratorType switch
            {
                AcceleratorType.CPU => "CPU",
                AcceleratorType.Cuda => "CUDA",
                AcceleratorType.OpenCL => "OpenCL",
                _ => "Unknown"
            },
            DeviceIndex = index,
            ComputeUnits = device.NumMultiprocessors,
            MaxThreadsPerGroup = device.MaxNumThreadsPerGroup,
            MaxSharedMemoryPerGroup = device.MaxSharedMemoryPerGroup,
            MemorySize = device.MemorySize,
            WarpSize = device.WarpSize,
            ClockRate = device.ClockRate,
        };
    }
}
```

**Step 3: Create BenchmarkResult**

```csharp
namespace GpuBench.Models;

public enum VerificationStatus
{
    NotApplicable,
    Passed,
    Failed,
    Reference
}

public sealed class BenchmarkResult
{
    public required string SuiteName { get; init; }
    public required string BenchmarkName { get; init; }
    public required string DeviceName { get; init; }
    public required string MetricName { get; init; }
    public required string Unit { get; init; }
    public double Best { get; set; }
    public double Average { get; set; }
    public double Worst { get; set; }
    public double StdDev { get; set; }
    public VerificationStatus Verification { get; set; } = VerificationStatus.NotApplicable;
    public string? ErrorMessage { get; set; }
    public bool IsError => ErrorMessage != null;

    public void ComputeStats(List<double> measurements)
    {
        if (measurements.Count == 0) return;
        Best = measurements.Max();
        Worst = measurements.Min();
        Average = measurements.Average();
        if (measurements.Count > 1)
        {
            var avg = Average;
            StdDev = Math.Sqrt(measurements.Sum(x => (x - avg) * (x - avg)) / (measurements.Count - 1));
        }
    }

    /// <summary>
    /// For latency measurements where lower is better.
    /// </summary>
    public void ComputeStatsLowerIsBetter(List<double> measurements)
    {
        if (measurements.Count == 0) return;
        Best = measurements.Min();
        Worst = measurements.Max();
        Average = measurements.Average();
        if (measurements.Count > 1)
        {
            var avg = Average;
            StdDev = Math.Sqrt(measurements.Sum(x => (x - avg) * (x - avg)) / (measurements.Count - 1));
        }
    }
}
```

**Step 4: Build**

Run: `dotnet build GpuBench/GpuBench.csproj`
Expected: Build succeeded.

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add models for options, device profiles, and results"
```

---

### Task 3: IBenchmarkSuite Interface and Device Enumeration

**Files:**
- Create: `GpuBench/Benchmarks/IBenchmarkSuite.cs`
- Create: `GpuBench/Benchmarks/DeviceInfo.cs`
- Modify: `GpuBench/BenchmarkCommand.cs`

**Step 1: Create IBenchmarkSuite**

```csharp
using GpuBench.Models;
using GpuBench.Rendering;
using ILGPU.Runtime;

namespace GpuBench.Benchmarks;

public interface IBenchmarkSuite
{
    string Name { get; }
    string Description { get; }
    bool SupportsDevice(DeviceProfile device);
    List<BenchmarkResult> Run(IReadOnlyList<Accelerator> accelerators, IReadOnlyList<DeviceProfile> profiles, BenchmarkOptions options);
}
```

**Step 2: Create DeviceInfo**

This isn't a benchmark suite — it's a utility that displays device info. Create it as a static helper:

```csharp
using GpuBench.Models;
using Spectre.Console;

namespace GpuBench.Benchmarks;

public static class DeviceInfo
{
    public static void Display(IReadOnlyList<DeviceProfile> profiles)
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]Detected Compute Devices[/]");

        table.AddColumn(new TableColumn("[bold]#[/]").RightAligned());
        table.AddColumn(new TableColumn("[bold]Name[/]"));
        table.AddColumn(new TableColumn("[bold]Type[/]"));
        table.AddColumn(new TableColumn("[bold]Compute Units[/]").RightAligned());
        table.AddColumn(new TableColumn("[bold]Max Threads/Group[/]").RightAligned());
        table.AddColumn(new TableColumn("[bold]Shared Mem/Group[/]").RightAligned());
        table.AddColumn(new TableColumn("[bold]Memory[/]").RightAligned());
        table.AddColumn(new TableColumn("[bold]Warp Size[/]").RightAligned());
        table.AddColumn(new TableColumn("[bold]Clock (MHz)[/]").RightAligned());

        foreach (var p in profiles)
        {
            table.AddRow(
                $"[{p.Color}]{p.DeviceIndex}[/]",
                $"[{p.Color} bold]{Markup.Escape(p.Name)}[/]",
                $"[{p.Color}]{p.DeviceType}[/]",
                $"[{p.Color}]{p.ComputeUnits}[/]",
                $"[{p.Color}]{p.MaxThreadsPerGroup:N0}[/]",
                $"[{p.Color}]{FormatBytes(p.MaxSharedMemoryPerGroup)}[/]",
                $"[{p.Color}]{FormatBytes(p.MemorySize)}[/]",
                $"[{p.Color}]{p.WarpSize}[/]",
                $"[{p.Color}]{p.ClockRate}[/]"
            );
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    private static string FormatBytes(long bytes)
    {
        if (bytes >= 1L << 30) return $"{bytes / (double)(1L << 30):F1} GB";
        if (bytes >= 1L << 20) return $"{bytes / (double)(1L << 20):F1} MB";
        if (bytes >= 1L << 10) return $"{bytes / (double)(1L << 10):F1} KB";
        return $"{bytes} B";
    }
}
```

**Step 3: Update BenchmarkCommand to enumerate devices and display info**

Replace the `Execute` method in `BenchmarkCommand.cs` to:
1. Create ILGPU context
2. Enumerate all devices
3. Create accelerators (filtered by `--device` if specified)
4. Build DeviceProfile list
5. Display device table
6. If `--list`, stop here
7. Otherwise, run requested suites

```csharp
using GpuBench.Benchmarks;
using GpuBench.Models;
using ILGPU;
using ILGPU.Runtime;
using Spectre.Console;
using Spectre.Console.Cli;
using System.ComponentModel;

namespace GpuBench;

// ... BenchmarkSettings unchanged ...

public sealed class BenchmarkCommand : Command<BenchmarkSettings>
{
    public override int Execute(CommandContext context, BenchmarkSettings settings)
    {
        var options = BenchmarkOptions.FromSettings(settings);

        AnsiConsole.Write(new Rule("[bold blue]GpuBench v1.0[/]").RuleStyle("blue"));
        AnsiConsole.WriteLine();

        using var ilContext = Context.Create(builder =>
        {
            builder.Default();
            builder.EnableAlgorithms();
        });

        // Enumerate and filter devices
        var allDevices = ilContext.Devices;
        var accelerators = new List<Accelerator>();
        var profiles = new List<DeviceProfile>();
        int index = 0;

        foreach (var device in allDevices)
        {
            if (!MatchesFilter(device, options.DeviceFilter, index))
            {
                index++;
                continue;
            }

            var accel = device.CreateAccelerator(ilContext);
            var profile = DeviceProfile.FromAccelerator(accel, index);
            accelerators.Add(accel);
            profiles.Add(profile);
            index++;
        }

        if (profiles.Count == 0)
        {
            AnsiConsole.MarkupLine("[red]No matching devices found.[/]");
            return 1;
        }

        AnsiConsole.MarkupLine($"[bold]Detected {profiles.Count} compute device(s)[/]");
        AnsiConsole.WriteLine();
        DeviceInfo.Display(profiles);

        if (options.ListOnly)
        {
            foreach (var a in accelerators) a.Dispose();
            return 0;
        }

        // Run suites
        var allResults = new List<BenchmarkResult>();
        var suites = CreateSuites();

        foreach (var suite in suites)
        {
            if (options.Suite != null && !suite.Name.Equals(options.Suite, StringComparison.OrdinalIgnoreCase))
                continue;

            AnsiConsole.Write(new Rule($"[bold]{Markup.Escape(suite.Name)}[/] — {Markup.Escape(suite.Description)}").RuleStyle("dim"));
            AnsiConsole.WriteLine();

            try
            {
                var results = suite.Run(accelerators, profiles, options);
                allResults.AddRange(results);
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Suite '{Markup.Escape(suite.Name)}' failed: {Markup.Escape(ex.Message)}[/]");
            }

            AnsiConsole.WriteLine();
        }

        // TODO: Render summary table
        // TODO: Export results

        foreach (var a in accelerators) a.Dispose();
        return 0;
    }

    private static bool MatchesFilter(Device device, string? filter, int index)
    {
        if (filter == null) return true;
        if (int.TryParse(filter, out int idx)) return idx == index;
        if (filter == "cpu") return device.AcceleratorType == AcceleratorType.CPU;
        if (filter == "cuda") return device.AcceleratorType == AcceleratorType.Cuda;
        if (filter == "opencl") return device.AcceleratorType == AcceleratorType.OpenCL;
        return device.Name.Contains(filter, StringComparison.OrdinalIgnoreCase);
    }

    private static List<IBenchmarkSuite> CreateSuites() =>
    [
        // Suites will be added here as they are implemented
    ];
}
```

**Step 4: Build and test**

Run: `dotnet build GpuBench/GpuBench.csproj`
Run: `dotnet run --project GpuBench -- --list`
Expected: Shows detected devices table.

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: device enumeration with Spectre.Console table display"
```

---

### Task 4: Compute Throughput — Kernels

**Files:**
- Create: `GpuBench/Kernels/ComputeKernels.cs`

**Step 1: Write compute kernels**

ILGPU kernels are static methods. Each kernel performs a known number of FMA operations per thread so we can calculate GFLOPS from thread count * ops per thread / time.

```csharp
using ILGPU;
using ILGPU.Runtime;

namespace GpuBench.Kernels;

public static class ComputeKernels
{
    /// <summary>
    /// FP32 FMA throughput kernel. Each thread does 1000 multiply-adds.
    /// Total FLOPs per thread = 2000 (1 mul + 1 add per FMA = 2 FLOPs).
    /// </summary>
    public const int FmaOpsPerThread = 1000;
    public const int FlopsPerFma = 2;

    public static void Fp32FmaKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> output)
    {
        float val = index.X * 0.001f;
        for (int i = 0; i < FmaOpsPerThread; i++)
            val = val * 1.0001f + 0.0001f;
        output[index] = val;
    }

    public static void Fp64FmaKernel(Index1D index, ArrayView1D<double, Stride1D.Dense> output)
    {
        double val = index.X * 0.001;
        for (int i = 0; i < FmaOpsPerThread; i++)
            val = val * 1.0001 + 0.0001;
        output[index] = val;
    }

    public static void Int32FmaKernel(Index1D index, ArrayView1D<int, Stride1D.Dense> output)
    {
        int val = index.X;
        for (int i = 0; i < FmaOpsPerThread; i++)
            val = val * 3 + 7;
        output[index] = val;
    }

    /// <summary>
    /// Vector add: C[i] = A[i] + B[i]
    /// </summary>
    public static void VectorAddKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c)
    {
        c[index] = a[index] + b[index];
    }

    /// <summary>
    /// Vector FMA: C[i] = A[i] * B[i] + C[i]
    /// </summary>
    public static void VectorFmaKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c)
    {
        c[index] = a[index] * b[index] + c[index];
    }
}
```

**Step 2: Build**

Run: `dotnet build GpuBench/GpuBench.csproj`
Expected: Build succeeded.

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: add FP32/FP64/Int32 FMA and vector compute kernels"
```

---

### Task 5: Compute Throughput — Benchmark Suite

**Files:**
- Create: `GpuBench/Benchmarks/ComputeThroughput.cs`
- Modify: `GpuBench/BenchmarkCommand.cs` (add suite to list)

**Step 1: Implement ComputeThroughput suite**

```csharp
using System.Diagnostics;
using GpuBench.Kernels;
using GpuBench.Models;
using ILGPU;
using ILGPU.Runtime;
using Spectre.Console;

namespace GpuBench.Benchmarks;

public sealed class ComputeThroughput : IBenchmarkSuite
{
    public string Name => "compute";
    public string Description => "FP32/FP64/Int32 throughput and vector operations";
    public bool SupportsDevice(DeviceProfile device) => true;

    public List<BenchmarkResult> Run(IReadOnlyList<Accelerator> accelerators, IReadOnlyList<DeviceProfile> profiles, BenchmarkOptions options)
    {
        var results = new List<BenchmarkResult>();
        int[] threadCounts = options.Quick ? [1_000_000, 4_000_000] : [1_000_000, 4_000_000, 16_000_000];
        int vectorSize = options.Quick ? 16_000_000 : 64_000_000;

        // Reference results for verification
        float[]? refVectorAdd = null;
        float[]? refVectorFma = null;

        for (int d = 0; d < accelerators.Count; d++)
        {
            var accel = accelerators[d];
            var profile = profiles[d];

            AnsiConsole.MarkupLine($"  [{profile.Color}]{Markup.Escape(profile.Name)}[/]");

            // FP32
            results.Add(RunFp32(accel, profile, threadCounts, options));
            // FP64
            results.Add(RunFp64(accel, profile, threadCounts, options));
            // Int32
            results.Add(RunInt32(accel, profile, threadCounts, options));
            // Vector ops
            var (addResult, fmaResult, addOutput, fmaOutput) = RunVectorOps(accel, profile, vectorSize, options);
            results.Add(addResult);
            results.Add(fmaResult);

            // Verification
            if (d == 0)
            {
                refVectorAdd = addOutput;
                refVectorFma = fmaOutput;
                addResult.Verification = VerificationStatus.Reference;
                fmaResult.Verification = VerificationStatus.Reference;
            }
            else
            {
                addResult.Verification = VerifyResults(refVectorAdd!, addOutput, 1e-2f) ? VerificationStatus.Passed : VerificationStatus.Failed;
                fmaResult.Verification = VerifyResults(refVectorFma!, fmaOutput, 1e-2f) ? VerificationStatus.Passed : VerificationStatus.Failed;
            }
        }

        RenderResults(results, profiles);
        return results;
    }

    private BenchmarkResult RunFp32(Accelerator accel, DeviceProfile profile, int[] threadCounts, BenchmarkOptions options)
    {
        var result = new BenchmarkResult { SuiteName = "compute", BenchmarkName = "FP32 Throughput", DeviceName = profile.Name, MetricName = "GFLOPS", Unit = "GFLOPS" };
        try
        {
            var kernel = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(ComputeKernels.Fp32FmaKernel);
            var measurements = new List<double>();
            int bestThreadCount = threadCounts[^1];

            // Use largest thread count that fits in memory
            long requiredBytes = (long)bestThreadCount * sizeof(float);
            if (requiredBytes > profile.MemorySize * 0.8 && !profile.IsCpu)
            {
                bestThreadCount = threadCounts.LastOrDefault(t => (long)t * sizeof(float) <= profile.MemorySize * 0.8);
                if (bestThreadCount == 0) { result.ErrorMessage = "Insufficient memory"; return result; }
            }

            using var buffer = accel.Allocate1D<float>(bestThreadCount);

            // Warmup
            for (int w = 0; w < options.WarmupIterations; w++)
            {
                kernel(bestThreadCount, buffer.View);
                accel.Synchronize();
            }

            // Timed runs
            var sw = new Stopwatch();
            for (int i = 0; i < options.Iterations; i++)
            {
                sw.Restart();
                kernel(bestThreadCount, buffer.View);
                accel.Synchronize();
                sw.Stop();
                double seconds = sw.Elapsed.TotalSeconds;
                double flops = (double)bestThreadCount * ComputeKernels.FmaOpsPerThread * ComputeKernels.FlopsPerFma;
                measurements.Add(flops / seconds / 1e9);
            }
            result.ComputeStats(measurements);
            AnsiConsole.MarkupLine($"    FP32: [bold]{result.Best:N1}[/] GFLOPS (best of {options.Iterations})");
        }
        catch (Exception ex)
        {
            result.ErrorMessage = ex.Message;
            AnsiConsole.MarkupLine($"    FP32: [red]FAILED — {Markup.Escape(ex.Message)}[/]");
        }
        return result;
    }

    private BenchmarkResult RunFp64(Accelerator accel, DeviceProfile profile, int[] threadCounts, BenchmarkOptions options)
    {
        var result = new BenchmarkResult { SuiteName = "compute", BenchmarkName = "FP64 Throughput", DeviceName = profile.Name, MetricName = "GFLOPS", Unit = "GFLOPS" };
        try
        {
            var kernel = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>>(ComputeKernels.Fp64FmaKernel);
            var measurements = new List<double>();
            int bestThreadCount = threadCounts[^1];

            long requiredBytes = (long)bestThreadCount * sizeof(double);
            if (requiredBytes > profile.MemorySize * 0.8 && !profile.IsCpu)
            {
                bestThreadCount = threadCounts.LastOrDefault(t => (long)t * sizeof(double) <= profile.MemorySize * 0.8);
                if (bestThreadCount == 0) { result.ErrorMessage = "Insufficient memory"; return result; }
            }

            using var buffer = accel.Allocate1D<double>(bestThreadCount);

            for (int w = 0; w < options.WarmupIterations; w++)
            {
                kernel(bestThreadCount, buffer.View);
                accel.Synchronize();
            }

            var sw = new Stopwatch();
            for (int i = 0; i < options.Iterations; i++)
            {
                sw.Restart();
                kernel(bestThreadCount, buffer.View);
                accel.Synchronize();
                sw.Stop();
                double seconds = sw.Elapsed.TotalSeconds;
                double flops = (double)bestThreadCount * ComputeKernels.FmaOpsPerThread * ComputeKernels.FlopsPerFma;
                measurements.Add(flops / seconds / 1e9);
            }
            result.ComputeStats(measurements);
            AnsiConsole.MarkupLine($"    FP64: [bold]{result.Best:N1}[/] GFLOPS (best of {options.Iterations})");
        }
        catch (Exception ex)
        {
            result.ErrorMessage = ex.Message;
            AnsiConsole.MarkupLine($"    FP64: [red]FAILED — {Markup.Escape(ex.Message)}[/]");
        }
        return result;
    }

    private BenchmarkResult RunInt32(Accelerator accel, DeviceProfile profile, int[] threadCounts, BenchmarkOptions options)
    {
        var result = new BenchmarkResult { SuiteName = "compute", BenchmarkName = "Int32 Throughput", DeviceName = profile.Name, MetricName = "GOPS", Unit = "GOPS" };
        try
        {
            var kernel = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>>(ComputeKernels.Int32FmaKernel);
            var measurements = new List<double>();
            int bestThreadCount = threadCounts[^1];

            long requiredBytes = (long)bestThreadCount * sizeof(int);
            if (requiredBytes > profile.MemorySize * 0.8 && !profile.IsCpu)
            {
                bestThreadCount = threadCounts.LastOrDefault(t => (long)t * sizeof(int) <= profile.MemorySize * 0.8);
                if (bestThreadCount == 0) { result.ErrorMessage = "Insufficient memory"; return result; }
            }

            using var buffer = accel.Allocate1D<int>(bestThreadCount);

            for (int w = 0; w < options.WarmupIterations; w++)
            {
                kernel(bestThreadCount, buffer.View);
                accel.Synchronize();
            }

            var sw = new Stopwatch();
            for (int i = 0; i < options.Iterations; i++)
            {
                sw.Restart();
                kernel(bestThreadCount, buffer.View);
                accel.Synchronize();
                sw.Stop();
                double seconds = sw.Elapsed.TotalSeconds;
                // 2 ops per iteration (mul + add)
                double ops = (double)bestThreadCount * ComputeKernels.FmaOpsPerThread * 2;
                measurements.Add(ops / seconds / 1e9);
            }
            result.ComputeStats(measurements);
            AnsiConsole.MarkupLine($"    Int32: [bold]{result.Best:N1}[/] GOPS (best of {options.Iterations})");
        }
        catch (Exception ex)
        {
            result.ErrorMessage = ex.Message;
            AnsiConsole.MarkupLine($"    Int32: [red]FAILED — {Markup.Escape(ex.Message)}[/]");
        }
        return result;
    }

    private (BenchmarkResult addResult, BenchmarkResult fmaResult, float[] addOutput, float[] fmaOutput) RunVectorOps(
        Accelerator accel, DeviceProfile profile, int vectorSize, BenchmarkOptions options)
    {
        var addResult = new BenchmarkResult { SuiteName = "compute", BenchmarkName = "Vector Add", DeviceName = profile.Name, MetricName = "GB/s", Unit = "GB/s" };
        var fmaResult = new BenchmarkResult { SuiteName = "compute", BenchmarkName = "Vector FMA", DeviceName = profile.Name, MetricName = "GFLOPS", Unit = "GFLOPS" };
        float[] addOutput = [];
        float[] fmaOutput = [];

        try
        {
            // Check memory: need 3 buffers of vectorSize floats
            long requiredBytes = (long)vectorSize * sizeof(float) * 3;
            if (requiredBytes > profile.MemorySize * 0.8 && !profile.IsCpu)
            {
                vectorSize = (int)(profile.MemorySize * 0.8 / sizeof(float) / 3);
            }

            var addKernel = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ComputeKernels.VectorAddKernel);
            var fmaKernel = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(ComputeKernels.VectorFmaKernel);

            // Initialize host data
            var hostA = new float[vectorSize];
            var hostB = new float[vectorSize];
            var hostC = new float[vectorSize];
            var rng = new Random(42);
            for (int i = 0; i < vectorSize; i++)
            {
                hostA[i] = (float)(rng.NextDouble() * 2 - 1);
                hostB[i] = (float)(rng.NextDouble() * 2 - 1);
                hostC[i] = (float)(rng.NextDouble() * 2 - 1);
            }

            using var bufA = accel.Allocate1D<float>(vectorSize);
            using var bufB = accel.Allocate1D<float>(vectorSize);
            using var bufC = accel.Allocate1D<float>(vectorSize);

            bufA.CopyFromCPU(hostA);
            bufB.CopyFromCPU(hostB);
            accel.Synchronize();

            // Vector Add
            {
                // Warmup
                for (int w = 0; w < options.WarmupIterations; w++)
                {
                    addKernel(vectorSize, bufA.View, bufB.View, bufC.View);
                    accel.Synchronize();
                }

                var measurements = new List<double>();
                var sw = new Stopwatch();
                for (int i = 0; i < options.Iterations; i++)
                {
                    sw.Restart();
                    addKernel(vectorSize, bufA.View, bufB.View, bufC.View);
                    accel.Synchronize();
                    sw.Stop();
                    // 3 arrays accessed * vectorSize * 4 bytes each
                    double bytes = (double)vectorSize * sizeof(float) * 3;
                    measurements.Add(bytes / sw.Elapsed.TotalSeconds / 1e9);
                }
                addResult.ComputeStats(measurements);
                addOutput = bufC.GetAsArray1D();
                AnsiConsole.MarkupLine($"    Vec Add: [bold]{addResult.Best:N1}[/] GB/s");
            }

            // Vector FMA
            {
                bufC.CopyFromCPU(hostC);
                accel.Synchronize();

                for (int w = 0; w < options.WarmupIterations; w++)
                {
                    bufC.CopyFromCPU(hostC);
                    fmaKernel(vectorSize, bufA.View, bufB.View, bufC.View);
                    accel.Synchronize();
                }

                var measurements = new List<double>();
                var sw = new Stopwatch();
                for (int i = 0; i < options.Iterations; i++)
                {
                    bufC.CopyFromCPU(hostC);
                    accel.Synchronize();
                    sw.Restart();
                    fmaKernel(vectorSize, bufA.View, bufB.View, bufC.View);
                    accel.Synchronize();
                    sw.Stop();
                    // 2 FLOPs per element (mul + add)
                    double flops = (double)vectorSize * 2;
                    measurements.Add(flops / sw.Elapsed.TotalSeconds / 1e9);
                }
                fmaResult.ComputeStats(measurements);
                fmaOutput = bufC.GetAsArray1D();
                AnsiConsole.MarkupLine($"    Vec FMA: [bold]{fmaResult.Best:N1}[/] GFLOPS");
            }
        }
        catch (Exception ex)
        {
            addResult.ErrorMessage ??= ex.Message;
            fmaResult.ErrorMessage ??= ex.Message;
            AnsiConsole.MarkupLine($"    Vector ops: [red]FAILED — {Markup.Escape(ex.Message)}[/]");
        }

        return (addResult, fmaResult, addOutput, fmaOutput);
    }

    private static bool VerifyResults(float[] reference, float[] test, float epsilon)
    {
        if (reference.Length != test.Length || reference.Length == 0) return false;
        // Check a sample of elements (every 1000th)
        for (int i = 0; i < reference.Length; i += 1000)
        {
            if (MathF.Abs(reference[i] - test[i]) > epsilon * MathF.Max(1f, MathF.Abs(reference[i])))
                return false;
        }
        return true;
    }

    private void RenderResults(List<BenchmarkResult> results, IReadOnlyList<DeviceProfile> profiles)
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]Compute Throughput[/]");

        table.AddColumn(new TableColumn("[bold]Benchmark[/]"));
        foreach (var p in profiles)
            table.AddColumn(new TableColumn($"[{p.Color} bold]{Markup.Escape(p.Name)}[/]").RightAligned());

        var benchmarks = results.Select(r => r.BenchmarkName).Distinct().ToList();
        foreach (var bench in benchmarks)
        {
            var row = new List<string> { $"[bold]{Markup.Escape(bench)}[/]" };
            var benchResults = results.Where(r => r.BenchmarkName == bench).ToList();
            double bestVal = benchResults.Where(r => !r.IsError).Select(r => r.Best).DefaultIfEmpty(0).Max();

            foreach (var p in profiles)
            {
                var r = benchResults.FirstOrDefault(r => r.DeviceName == p.Name);
                if (r == null || r.IsError)
                {
                    row.Add(r?.ErrorMessage != null ? $"[red]FAILED[/]" : "—");
                }
                else
                {
                    string verif = r.Verification switch
                    {
                        VerificationStatus.Passed => " [green]✓[/]",
                        VerificationStatus.Failed => " [red]⚠[/]",
                        VerificationStatus.Reference => " [dim]ref[/]",
                        _ => ""
                    };
                    string bold = r.Best == bestVal ? "bold " : "";
                    row.Add($"[{bold}{p.Color}]{r.Best:N1}[/] {r.Unit}{verif}");
                }
            }
            table.AddRow(row.ToArray());
        }

        AnsiConsole.Write(table);
    }
}
```

**Step 2: Add suite to BenchmarkCommand.CreateSuites()**

```csharp
private static List<IBenchmarkSuite> CreateSuites() =>
[
    new ComputeThroughput(),
];
```

**Step 3: Build and test**

Run: `dotnet build GpuBench/GpuBench.csproj`
Run: `dotnet run --project GpuBench -- --suite compute --quick`
Expected: Shows compute throughput results for all devices.

**Step 4: Commit**

```bash
git add -A && git commit -m "feat: compute throughput benchmarks (FP32/FP64/Int32/Vector)"
```

---

### Task 6: Matrix Multiplication — Kernels

**Files:**
- Create: `GpuBench/Kernels/MatMulKernels.cs`

**Step 1: Write matmul kernels**

```csharp
using ILGPU;
using ILGPU.Runtime;

namespace GpuBench.Kernels;

public static class MatMulKernels
{
    public const int TileSize = 16;

    /// <summary>
    /// Naive matmul: one thread computes one element of C = A * B.
    /// A, B, C are square matrices stored in row-major 1D arrays.
    /// </summary>
    public static void NaiveMatMulKernel(
        Index2D index,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> c,
        int n)
    {
        int row = index.X;
        int col = index.Y;
        if (row >= n || col >= n) return;

        float sum = 0;
        for (int k = 0; k < n; k++)
            sum += a[row * n + k] * b[k * n + col];
        c[row * n + col] = sum;
    }

    /// <summary>
    /// Tiled matmul using shared memory with 16x16 tiles.
    /// </summary>
    public static void TiledMatMulKernel(
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> c,
        int n)
    {
        var tileA = ILGPU.SharedMemory.Allocate2D<float, Stride2D.DenseX>(new Index2D(TileSize, TileSize), new Stride2D.DenseX(TileSize));
        var tileB = ILGPU.SharedMemory.Allocate2D<float, Stride2D.DenseX>(new Index2D(TileSize, TileSize), new Stride2D.DenseX(TileSize));

        int row = Grid.GlobalIndex.X;
        int col = Grid.GlobalIndex.Y;
        int localRow = Group.IdxX;
        int localCol = Group.IdxY;

        float sum = 0;
        int numTiles = (n + TileSize - 1) / TileSize;

        for (int t = 0; t < numTiles; t++)
        {
            int aCol = t * TileSize + localCol;
            int bRow = t * TileSize + localRow;

            tileA[localRow, localCol] = (row < n && aCol < n) ? a[row * n + aCol] : 0f;
            tileB[localRow, localCol] = (bRow < n && col < n) ? b[bRow * n + col] : 0f;

            Group.Barrier();

            for (int k = 0; k < TileSize; k++)
                sum += tileA[localRow, k] * tileB[k, localCol];

            Group.Barrier();
        }

        if (row < n && col < n)
            c[row * n + col] = sum;
    }
}
```

**Step 2: Build**

Run: `dotnet build GpuBench/GpuBench.csproj`
Expected: Build succeeded.

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: add naive and tiled matmul kernels"
```

---

### Task 7: Matrix Multiplication — Benchmark Suite

**Files:**
- Create: `GpuBench/Benchmarks/MatrixMultiply.cs`
- Modify: `GpuBench/BenchmarkCommand.cs` (add suite)

**Step 1: Implement MatrixMultiply suite**

The suite runs naive and tiled kernels at multiple sizes, performs scaling analysis, and cross-device verification. CPU is limited to size <= 2048 unless `--full`.

Key details:
- GFLOPS for NxN matmul = 2 * N^3 / time / 1e9
- Use `LoadAutoGroupedStreamKernel` for naive (uses Index2D)
- For tiled, use explicit group size of (TileSize, TileSize) = (16, 16)
- Verify by comparing output matrices from different devices against first device reference
- Display results in a Spectre table with naive vs tiled, speedup column

The implementation follows the same pattern as ComputeThroughput:
- For each device, for each size:
  - Allocate A, B, C as 1D buffers of N*N floats
  - Initialize A, B with deterministic random data (seed 42)
  - Run warmup, then timed iterations
  - Calculate GFLOPS
  - Store reference output from first device for verification
- Render table with sizes as rows, device+kernel combinations as columns

**Step 2: Add to CreateSuites()**

```csharp
new MatrixMultiply(),
```

**Step 3: Build and test**

Run: `dotnet build GpuBench/GpuBench.csproj`
Run: `dotnet run --project GpuBench -- --suite matmul --quick`
Expected: Shows matmul results with naive vs tiled comparison.

**Step 4: Commit**

```bash
git add -A && git commit -m "feat: matrix multiplication benchmarks with naive + tiled kernels"
```

---

### Task 8: Memory Bandwidth Suite

**Files:**
- Create: `GpuBench/Benchmarks/MemoryBandwidth.cs`
- Modify: `GpuBench/BenchmarkCommand.cs` (add suite)

**Step 1: Implement MemoryBandwidth suite**

Tests H→D, D→H, and D→D transfers at sizes 1MB, 4MB, 16MB, 64MB, 256MB, 1GB (quick mode skips 256MB and 1GB).

Key details:
- `SupportsDevice` returns false for CPU
- H→D: allocate host array, time `buffer.CopyFromCPU(hostArray)` + `accel.Synchronize()`
- D→H: time `buffer.GetAsArray1D()` (includes sync)
- D→D: allocate two device buffers, time `src.CopyTo(dst)` + sync
- Skip sizes that exceed 80% of device memory
- Report bandwidth in GB/s = bytes / seconds / 1e9
- 5 runs each (3 in quick mode), report best/avg/worst
- Render per-test-type Spectre table with all devices side by side

**Step 2: Add to CreateSuites() and build**

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: memory bandwidth benchmarks (H→D, D→H, D→D)"
```

---

### Task 9: Latency Tests Suite

**Files:**
- Create: `GpuBench/Benchmarks/LatencyTests.cs`
- Modify: `GpuBench/BenchmarkCommand.cs` (add suite)

**Step 1: Implement LatencyTests suite**

Three tests:

**Kernel launch overhead:**
- Create an empty kernel: `static void EmptyKernel(Index1D index) { }`
- Launch it, synchronize, measure round-trip time
- 100 launches (50 quick), report min/avg/max/p99 in microseconds

**Memory allocation latency:**
- Time `accel.Allocate1D<float>(size)` for sizes 1K, 64K, 1M, 16M elements
- Include dispose in timing
- 50 iterations (25 quick)

**Synchronization overhead:**
- Time `accel.Synchronize()` with no pending work (idle sync)
- Time `accel.Synchronize()` after launching empty kernel (post-kernel sync)
- 100 iterations, report in microseconds

All timings use `Stopwatch` with microsecond precision. Results rendered in a table with μs units.

**Step 2: Add to CreateSuites() and build**

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: latency benchmarks (kernel launch, alloc, sync overhead)"
```

---

### Task 10: Memory Patterns — Kernels

**Files:**
- Create: `GpuBench/Kernels/MemoryKernels.cs`

**Step 1: Write memory pattern kernels**

```csharp
using ILGPU;
using ILGPU.Runtime;

namespace GpuBench.Kernels;

public static class MemoryKernels
{
    /// <summary>
    /// Sequential read: thread i reads element i.
    /// </summary>
    public static void SequentialReadKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output)
    {
        output[index] = input[index];
    }

    /// <summary>
    /// Random read: thread i reads element from shuffled index array.
    /// </summary>
    public static void RandomReadKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<int, Stride1D.Dense> indices, ArrayView1D<float, Stride1D.Dense> output)
    {
        output[index] = input[indices[index]];
    }

    /// <summary>
    /// Strided read: thread i reads element i * stride.
    /// </summary>
    public static void StridedReadKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int stride, int maxIndex)
    {
        int readIdx = (index.X * stride) % maxIndex;
        output[index] = input[readIdx];
    }

    /// <summary>
    /// Global memory repeated read: reads same element from global memory N times.
    /// </summary>
    public static void GlobalRepeatedReadKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int repeats)
    {
        float sum = 0;
        for (int i = 0; i < repeats; i++)
            sum += input[index];
        output[index] = sum;
    }

    /// <summary>
    /// Shared memory repeated read: loads into shared memory once, reads N times.
    /// Group size must match shared allocation.
    /// </summary>
    public static void SharedRepeatedReadKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> input, ArrayView1D<float, Stride1D.Dense> output, int repeats)
    {
        var shared = ILGPU.SharedMemory.Allocate1D<float>(1024);
        int localIdx = Group.IdxX;
        int globalIdx = Grid.GlobalIndex.X;

        if (localIdx < 1024 && globalIdx < input.Length)
            shared[localIdx] = input[globalIdx];
        Group.Barrier();

        float sum = 0;
        if (localIdx < 1024)
        {
            for (int i = 0; i < repeats; i++)
                sum += shared[localIdx];
        }

        if (globalIdx < output.Length)
            output[globalIdx] = sum;
    }
}
```

**Step 2: Build and commit**

```bash
git add -A && git commit -m "feat: add memory pattern test kernels"
```

---

### Task 11: Memory Patterns — Benchmark Suite

**Files:**
- Create: `GpuBench/Benchmarks/MemoryPatterns.cs`
- Modify: `GpuBench/BenchmarkCommand.cs` (add suite)

**Step 1: Implement MemoryPatterns suite**

Three test categories:

**Sequential vs Random:**
- 64M float buffer (16M quick), shuffled index array for random
- Measure effective bandwidth (GB/s) for each pattern
- Show ratio (sequential / random)

**Shared vs Global Memory:**
- Repeated reads (100 repeats) from global vs shared memory
- Measure effective bandwidth
- Show ratio (shared / global)

**Coalesced vs Strided:**
- Coalesced = stride 1 (adjacent threads read adjacent memory)
- Strided with stride 2, 4, 8, 16, 32
- Measure bandwidth at each stride, show degradation

Verification: compare sequential and random read outputs against each other within epsilon for cross-device checks.

**Step 2: Add to CreateSuites() and build**

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: memory access pattern benchmarks"
```

---

### Task 12: Results Renderer

**Files:**
- Create: `GpuBench/Rendering/ResultsRenderer.cs`

**Step 1: Implement ResultsRenderer**

Static class with methods:

- `RenderSummaryTable(List<BenchmarkResult> results, List<DeviceProfile> profiles)` — overall summary table at end of full run, showing best metric per suite per device
- `RenderBarChart(string title, Dictionary<string, double> deviceValues, string unit)` — Spectre bar chart for visual comparison
- `RenderWinnerPanel(string category, DeviceProfile winner, double value, string unit, List<(DeviceProfile device, double ratio)> comparisons)` — winner callout panel after each suite
- `RenderOverallChampion(List<BenchmarkResult> results, List<DeviceProfile> profiles)` — overall winner across all suites

Summary table columns: Benchmark | Device1 | Device2 | ... with bold best values, right-aligned, thousands separators.

Bar chart uses `BreakdownChart` from Spectre.Console (since `BarChart` is limited). Color per device type.

Winner panel format:
```
╭─ 🏆 Compute Champion ──────────────────────╮
│  gfx1101 (OpenCL) — 8,432.1 GFLOPS FP32    │
│  186x faster than CPU, 7x faster than CUDA  │
╰─────────────────────────────────────────────╯
```

**Step 2: Integrate into BenchmarkCommand**

After all suites run, call `ResultsRenderer.RenderSummaryTable()` and `RenderOverallChampion()`. Add winner panels at the end of each suite's `Run` method.

**Step 3: Build and commit**

```bash
git add -A && git commit -m "feat: results renderer with summary tables, bar charts, winner panels"
```

---

### Task 13: Progress Tracker

**Files:**
- Create: `GpuBench/Rendering/ProgressTracker.cs`

**Step 1: Implement ProgressTracker**

Wrap `AnsiConsole.Status()` for per-benchmark status updates. The suites already print inline results, so the tracker provides:
- `Status(string message, Action action)` — shows spinner during action
- Colored device name in status messages

This is lightweight — suites already handle their own progress output. The tracker adds the spinner wrapper.

**Step 2: Integrate into suite Run methods where appropriate**

**Step 3: Build and commit**

```bash
git add -A && git commit -m "feat: progress tracker with status spinners"
```

---

### Task 14: Export Writer

**Files:**
- Create: `GpuBench/Rendering/ExportWriter.cs`

**Step 1: Implement ExportWriter**

Two methods:

`ExportJson(string path, List<BenchmarkResult> results, List<DeviceProfile> profiles)`:
- Serialize to JSON with System.Text.Json
- Include timestamp, system info (OS, machine name), device profiles, all results
- Pretty-printed with indentation

`ExportMarkdown(string path, List<BenchmarkResult> results, List<DeviceProfile> profiles)`:
- Generate Markdown tables grouped by suite
- Include device info table, all benchmark tables, winner notes
- Suitable for pasting into GitHub issues

**Step 2: Integrate into BenchmarkCommand**

After suites complete, if `options.ExportPath` is set:
- If path ends with `.json`, call `ExportJson`
- If path ends with `.md`, call `ExportMarkdown`
- Otherwise, warn and default to JSON

**Step 3: Build and test export**

Run: `dotnet run --project GpuBench -- --suite compute --quick --export results.json`
Run: `dotnet run --project GpuBench -- --suite compute --quick --export results.md`
Expected: Files created with correct content.

**Step 4: Commit**

```bash
git add -A && git commit -m "feat: JSON and Markdown export"
```

---

### Task 15: Integration and Polish

**Files:**
- Modify: `GpuBench/BenchmarkCommand.cs`
- All suite files (minor tweaks)

**Step 1: Wire up all suites in CreateSuites()**

Ensure order matches priority: compute, matmul, memory, latency, patterns.

**Step 2: Add header banner**

```csharp
AnsiConsole.Write(new FigletText("GpuBench").Color(Color.Blue));
```

Or a simpler panel:
```csharp
AnsiConsole.Write(new Panel("[bold blue]GpuBench v1.0[/]")
    .Border(BoxBorder.Rounded)
    .Header("[bold]GPU/CPU Compute Benchmark Suite[/]"));
```

**Step 3: Full integration test**

Run: `dotnet run --project GpuBench`
Expected: All suites run on all devices, summary table shown, no crashes.

Run: `dotnet run --project GpuBench -- --quick --export results.json`
Expected: Quick run with JSON export.

Run: `dotnet run --project GpuBench -- --list`
Expected: Device list only.

Run: `dotnet run --project GpuBench -- --device cpu --suite compute --quick`
Expected: CPU-only compute benchmarks.

**Step 4: Commit and push**

```bash
git add -A && git commit -m "feat: integration, polish, and full suite wiring"
git push
```

---

## Notes for Implementer

- **ILGPU Version Compatibility:** If ILGPU 1.5.x has API differences from what's shown (e.g., `SharedMemory.Allocate2D` signature), check the ILGPU docs/samples and adapt. The tiled matmul kernel's shared memory API is the most likely to need adjustment.
- **Tiled Matmul Kernel Loading:** The tiled kernel uses `Group.Barrier()` and `Grid.GlobalIndex`, so it must be loaded with explicit group dimensions using `LoadStreamKernel` and launched with `kernel((gridSize, groupSize), args...)` where `groupSize = (TileSize, TileSize)`.
- **.NET 9 SDK:** Must be installed first. Use `dotnet --version` to verify.
- **NuGet Versions:** Use `dotnet add package <name>` to get latest stable versions rather than hardcoding versions from this plan.
- **GPU Drivers:** The machine needs CUDA/OpenCL drivers installed for GPU devices to appear. If only CPU shows up, GPU benchmarks still work but only on CPU.
