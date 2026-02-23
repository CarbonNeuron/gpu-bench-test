using System.Diagnostics;
using GpuBench.Models;
using ILGPU;
using ILGPU.Runtime;
using Spectre.Console;

namespace GpuBench.Benchmarks;

public sealed class LatencyTests : IBenchmarkSuite
{
    public string Name => "latency";
    public string Description => "Kernel launch, allocation, and sync overhead";
    public bool SupportsDevice(DeviceProfile device) => true;

    private static void EmptyKernel(Index1D index, ArrayView1D<int, Stride1D.Dense> dummy) { }

    public List<BenchmarkResult> Run(
        IReadOnlyList<Accelerator> accelerators,
        IReadOnlyList<DeviceProfile> profiles,
        BenchmarkOptions options)
    {
        var allResults = new List<BenchmarkResult>();

        for (int di = 0; di < accelerators.Count; di++)
        {
            var accel = accelerators[di];
            var profile = profiles[di];

            if (!SupportsDevice(profile)) continue;

            AnsiConsole.MarkupLine($"  [{profile.Color} bold]{Markup.Escape(profile.Name)}[/]");

            // Test 1: Kernel Launch Overhead
            try
            {
                var result = RunKernelLaunchOverhead(accel, profile, options);
                allResults.Add(result);
                PrintInlineResult(result, profile);
            }
            catch (Exception ex)
            {
                var result = CreateErrorResult("Kernel Launch", profile, ex);
                allResults.Add(result);
                AnsiConsole.MarkupLine($"    [red]Kernel Launch: FAILED - {Markup.Escape(ex.Message)}[/]");
            }

            // Test 2: Memory Allocation Latency
            int[] allocSizes = [1024, 65536, 1_000_000, 16_000_000];
            string[] allocLabels = ["Alloc 1K", "Alloc 64K", "Alloc 1M", "Alloc 16M"];

            for (int si = 0; si < allocSizes.Length; si++)
            {
                try
                {
                    var result = RunAllocationLatency(accel, profile, options, allocSizes[si], allocLabels[si]);
                    allResults.Add(result);
                    PrintInlineResult(result, profile);
                }
                catch (Exception ex)
                {
                    var result = CreateErrorResult(allocLabels[si], profile, ex);
                    allResults.Add(result);
                    AnsiConsole.MarkupLine($"    [red]{Markup.Escape(allocLabels[si])}: FAILED - {Markup.Escape(ex.Message)}[/]");
                }
            }

            // Test 3a: Idle Sync Overhead
            try
            {
                var result = RunIdleSyncOverhead(accel, profile, options);
                allResults.Add(result);
                PrintInlineResult(result, profile);
            }
            catch (Exception ex)
            {
                var result = CreateErrorResult("Idle Sync", profile, ex);
                allResults.Add(result);
                AnsiConsole.MarkupLine($"    [red]Idle Sync: FAILED - {Markup.Escape(ex.Message)}[/]");
            }

            // Test 3b: Post-Kernel Sync Overhead
            try
            {
                var result = RunPostKernelSyncOverhead(accel, profile, options);
                allResults.Add(result);
                PrintInlineResult(result, profile);
            }
            catch (Exception ex)
            {
                var result = CreateErrorResult("Post-Kernel Sync", profile, ex);
                allResults.Add(result);
                AnsiConsole.MarkupLine($"    [red]Post-Kernel Sync: FAILED - {Markup.Escape(ex.Message)}[/]");
            }

            AnsiConsole.WriteLine();
        }

        // Render summary table
        RenderSummaryTable(allResults, profiles);

        return allResults;
    }

    private BenchmarkResult RunKernelLaunchOverhead(
        Accelerator accel, DeviceProfile profile, BenchmarkOptions options)
    {
        var kernel = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>>(EmptyKernel);
        using var dummyBuf = accel.Allocate1D<int>(1);

        // Warmup
        for (int w = 0; w < 5; w++)
        {
            kernel(1, dummyBuf.View);
            accel.Synchronize();
        }

        int iterations = options.Quick ? 50 : 100;
        var measurements = new List<double>(iterations);

        for (int i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            kernel(1, dummyBuf.View);
            accel.Synchronize();
            sw.Stop();

            measurements.Add(sw.Elapsed.TotalMilliseconds * 1000.0); // microseconds
        }

        // Compute p99
        var sorted = measurements.OrderBy(x => x).ToList();
        int p99Index = (int)Math.Ceiling(sorted.Count * 0.99) - 1;
        p99Index = Math.Clamp(p99Index, 0, sorted.Count - 1);
        double p99 = sorted[p99Index];

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "Kernel Launch",
            DeviceName = profile.Name,
            MetricName = $"p99={p99:F1}",
            Unit = "\u00b5s",
        };
        result.ComputeStatsLowerIsBetter(measurements);
        return result;
    }

    private BenchmarkResult RunAllocationLatency(
        Accelerator accel, DeviceProfile profile, BenchmarkOptions options,
        int elementCount, string label)
    {
        int iterations = options.Quick ? 25 : 50;
        var measurements = new List<double>(iterations);

        // Warmup: one allocation/dispose cycle
        using (var warmup = accel.Allocate1D<float>(elementCount)) { }

        for (int i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            var buf = accel.Allocate1D<float>(elementCount);
            sw.Stop();
            buf.Dispose();

            measurements.Add(sw.Elapsed.TotalMilliseconds * 1000.0); // microseconds
        }

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = label,
            DeviceName = profile.Name,
            MetricName = $"{elementCount:N0} floats",
            Unit = "\u00b5s",
        };
        result.ComputeStatsLowerIsBetter(measurements);
        return result;
    }

    private BenchmarkResult RunIdleSyncOverhead(
        Accelerator accel, DeviceProfile profile, BenchmarkOptions options)
    {
        // Ensure no pending work
        accel.Synchronize();

        int iterations = options.Quick ? 50 : 100;
        var measurements = new List<double>(iterations);

        // Warmup
        for (int w = 0; w < 5; w++)
            accel.Synchronize();

        for (int i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            accel.Synchronize();
            sw.Stop();

            measurements.Add(sw.Elapsed.TotalMilliseconds * 1000.0); // microseconds
        }

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "Idle Sync",
            DeviceName = profile.Name,
            MetricName = "no pending work",
            Unit = "\u00b5s",
        };
        result.ComputeStatsLowerIsBetter(measurements);
        return result;
    }

    private BenchmarkResult RunPostKernelSyncOverhead(
        Accelerator accel, DeviceProfile profile, BenchmarkOptions options)
    {
        var kernel = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>>(EmptyKernel);
        using var dummyBuf = accel.Allocate1D<int>(1);

        // Warmup
        for (int w = 0; w < 5; w++)
        {
            kernel(1, dummyBuf.View);
            accel.Synchronize();
        }

        int iterations = options.Quick ? 50 : 100;
        var measurements = new List<double>(iterations);

        for (int i = 0; i < iterations; i++)
        {
            kernel(1, dummyBuf.View);
            var sw = Stopwatch.StartNew();
            accel.Synchronize();
            sw.Stop();

            measurements.Add(sw.Elapsed.TotalMilliseconds * 1000.0); // microseconds
        }

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "Post-Kernel Sync",
            DeviceName = profile.Name,
            MetricName = "after empty kernel",
            Unit = "\u00b5s",
        };
        result.ComputeStatsLowerIsBetter(measurements);
        return result;
    }

    private void RenderSummaryTable(List<BenchmarkResult> results, IReadOnlyList<DeviceProfile> profiles)
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]Latency Measurements[/]");

        table.AddColumn(new TableColumn("[bold]Test[/]"));
        foreach (var profile in profiles)
        {
            table.AddColumn(new TableColumn($"[{profile.Color} bold]{Markup.Escape(profile.Name)}[/]").RightAligned());
        }

        var benchmarkNames = new[]
        {
            "Kernel Launch",
            "Alloc 1K",
            "Alloc 64K",
            "Alloc 1M",
            "Alloc 16M",
            "Idle Sync",
            "Post-Kernel Sync"
        };

        foreach (var benchName in benchmarkNames)
        {
            var rowValues = new List<string>();
            rowValues.Add($"[bold]{Markup.Escape(benchName)} (\u00b5s)[/]");

            var deviceValues = new List<(int idx, double value, BenchmarkResult? result)>();

            foreach (var profile in profiles)
            {
                var result = results.FirstOrDefault(r =>
                    r.BenchmarkName == benchName && r.DeviceName == profile.Name);

                if (result == null)
                {
                    deviceValues.Add((-1, 0, null));
                }
                else if (result.IsError)
                {
                    deviceValues.Add((-1, 0, result));
                }
                else
                {
                    deviceValues.Add((deviceValues.Count, result.Best, result));
                }
            }

            // Find best (lowest) value across devices
            double bestValue = double.MaxValue;
            foreach (var dv in deviceValues)
            {
                if (dv.idx >= 0 && dv.value < bestValue)
                    bestValue = dv.value;
            }

            foreach (var dv in deviceValues)
            {
                if (dv.result == null)
                {
                    rowValues.Add("[dim]N/A[/]");
                }
                else if (dv.result.IsError)
                {
                    rowValues.Add("[red]FAILED[/]");
                }
                else
                {
                    string valueStr = $"{dv.value:F1}";
                    if (Math.Abs(dv.value - bestValue) < 0.01 && deviceValues.Count(v => v.idx >= 0) > 1)
                    {
                        rowValues.Add($"[bold]{valueStr}[/]");
                    }
                    else
                    {
                        rowValues.Add(valueStr);
                    }
                }
            }

            table.AddRow(rowValues.ToArray());
        }

        AnsiConsole.Write(table);
    }

    private static void PrintInlineResult(BenchmarkResult result, DeviceProfile profile)
    {
        if (result.IsError)
        {
            AnsiConsole.MarkupLine($"    [red]{Markup.Escape(result.BenchmarkName)}: FAILED[/]");
        }
        else
        {
            AnsiConsole.MarkupLine(
                $"    [{profile.Color}]{Markup.Escape(result.BenchmarkName)}: {result.Best:F1} {result.Unit} (avg {result.Average:F1}, max {result.Worst:F1})[/]");
        }
    }

    private BenchmarkResult CreateErrorResult(string benchmarkName, DeviceProfile profile, Exception ex)
    {
        return new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = benchmarkName,
            DeviceName = profile.Name,
            MetricName = "error",
            Unit = "\u00b5s",
            ErrorMessage = ex.Message,
        };
    }
}
