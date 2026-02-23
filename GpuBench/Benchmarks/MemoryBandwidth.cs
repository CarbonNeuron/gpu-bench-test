using System.Diagnostics;
using GpuBench.Models;
using ILGPU;
using ILGPU.Runtime;
using Spectre.Console;

namespace GpuBench.Benchmarks;

public sealed class MemoryBandwidth : IBenchmarkSuite
{
    public string Name => "memory";
    public string Description => "Host\u2194Device memory bandwidth";
    public bool SupportsDevice(DeviceProfile device) => !device.IsCpu;

    private static readonly (long sizeBytes, string label)[] AllSizes =
    [
        (1L * 1024 * 1024, "1 MB"),
        (4L * 1024 * 1024, "4 MB"),
        (16L * 1024 * 1024, "16 MB"),
        (64L * 1024 * 1024, "64 MB"),
        (256L * 1024 * 1024, "256 MB"),
        (1L * 1024 * 1024 * 1024, "1 GB"),
    ];

    private static readonly (long sizeBytes, string label)[] QuickSizes =
    [
        (1L * 1024 * 1024, "1 MB"),
        (4L * 1024 * 1024, "4 MB"),
        (16L * 1024 * 1024, "16 MB"),
        (64L * 1024 * 1024, "64 MB"),
    ];

    public List<BenchmarkResult> Run(
        IReadOnlyList<Accelerator> accelerators,
        IReadOnlyList<DeviceProfile> profiles,
        BenchmarkOptions options)
    {
        var allResults = new List<BenchmarkResult>();
        var sizes = options.Quick ? QuickSizes : AllSizes;

        // Check if any device is supported
        var supportedIndices = new List<int>();
        for (int di = 0; di < profiles.Count; di++)
        {
            if (SupportsDevice(profiles[di]))
                supportedIndices.Add(di);
        }

        if (supportedIndices.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]  No supported devices for memory bandwidth (CPU-only mode).[/]");
            return allResults;
        }

        // Run H->D, D->H, D->D for each supported device
        foreach (int di in supportedIndices)
        {
            var accel = accelerators[di];
            var profile = profiles[di];

            AnsiConsole.MarkupLine($"  [{profile.Color} bold]{Markup.Escape(profile.Name)}[/]");

            foreach (var (sizeBytes, label) in sizes)
            {
                if (ShouldSkipSize(sizeBytes, profile))
                {
                    allResults.Add(CreateSkippedResult("H\u2192D Transfer", label, profile));
                    allResults.Add(CreateSkippedResult("D\u2192H Transfer", label, profile));
                    allResults.Add(CreateSkippedResult("D\u2192D Copy", label, profile));
                    AnsiConsole.MarkupLine($"    [{profile.Color}]{Markup.Escape(label)}: skipped (exceeds 80% device memory)[/]");
                    continue;
                }

                int floatCount = (int)(sizeBytes / sizeof(float));

                // H->D
                try
                {
                    var result = RunHostToDevice(accel, profile, options, sizeBytes, floatCount, label);
                    allResults.Add(result);
                    PrintInlineResult("H\u2192D", result, profile, label);
                }
                catch (Exception ex)
                {
                    var result = CreateErrorResult("H\u2192D Transfer", label, profile, ex);
                    allResults.Add(result);
                    AnsiConsole.MarkupLine($"    [red]{Markup.Escape(label)} H\u2192D: FAILED - {Markup.Escape(ex.Message)}[/]");
                }

                // D->H
                try
                {
                    var result = RunDeviceToHost(accel, profile, options, sizeBytes, floatCount, label);
                    allResults.Add(result);
                    PrintInlineResult("D\u2192H", result, profile, label);
                }
                catch (Exception ex)
                {
                    var result = CreateErrorResult("D\u2192H Transfer", label, profile, ex);
                    allResults.Add(result);
                    AnsiConsole.MarkupLine($"    [red]{Markup.Escape(label)} D\u2192H: FAILED - {Markup.Escape(ex.Message)}[/]");
                }

                // D->D
                try
                {
                    var result = RunDeviceToDevice(accel, profile, options, sizeBytes, floatCount, label);
                    allResults.Add(result);
                    PrintInlineResult("D\u2192D", result, profile, label);
                }
                catch (Exception ex)
                {
                    var result = CreateErrorResult("D\u2192D Copy", label, profile, ex);
                    allResults.Add(result);
                    AnsiConsole.MarkupLine($"    [red]{Markup.Escape(label)} D\u2192D: FAILED - {Markup.Escape(ex.Message)}[/]");
                }
            }

            AnsiConsole.WriteLine();
        }

        // Render summary tables
        var supportedProfiles = supportedIndices.Select(i => profiles[i]).ToList();
        RenderBandwidthTable("Host \u2192 Device Bandwidth (GB/s)", "H\u2192D Transfer", allResults, supportedProfiles, sizes);
        RenderBandwidthTable("Device \u2192 Host Bandwidth (GB/s)", "D\u2192H Transfer", allResults, supportedProfiles, sizes);
        RenderBandwidthTable("Device \u2192 Device Bandwidth (GB/s)", "D\u2192D Copy", allResults, supportedProfiles, sizes);

        return allResults;
    }

    private BenchmarkResult RunHostToDevice(
        Accelerator accel, DeviceProfile profile, BenchmarkOptions options,
        long sizeBytes, int floatCount, string sizeLabel)
    {
        var hostArray = new float[floatCount];
        // Fill with some data
        for (int i = 0; i < floatCount; i++)
            hostArray[i] = i * 0.001f;

        using var buffer = accel.Allocate1D<float>(floatCount);

        // Warmup
        for (int w = 0; w < options.WarmupIterations; w++)
        {
            buffer.View.CopyFromCPU(hostArray);
            accel.Synchronize();
        }

        var measurements = new List<double>();

        for (int i = 0; i < options.Iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            buffer.View.CopyFromCPU(hostArray);
            accel.Synchronize();
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds;
            double gbps = sizeBytes / seconds / 1e9;
            measurements.Add(gbps);
        }

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "H\u2192D Transfer",
            DeviceName = profile.Name,
            MetricName = sizeLabel,
            Unit = "GB/s",
        };
        result.ComputeStats(measurements);
        return result;
    }

    private BenchmarkResult RunDeviceToHost(
        Accelerator accel, DeviceProfile profile, BenchmarkOptions options,
        long sizeBytes, int floatCount, string sizeLabel)
    {
        var hostArray = new float[floatCount];
        for (int i = 0; i < floatCount; i++)
            hostArray[i] = i * 0.001f;

        using var buffer = accel.Allocate1D<float>(floatCount);
        buffer.View.CopyFromCPU(hostArray);
        accel.Synchronize();

        // Warmup
        for (int w = 0; w < options.WarmupIterations; w++)
        {
            _ = buffer.GetAsArray1D();
            accel.Synchronize();
        }

        var measurements = new List<double>();

        for (int i = 0; i < options.Iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            _ = buffer.GetAsArray1D();
            accel.Synchronize();
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds;
            double gbps = sizeBytes / seconds / 1e9;
            measurements.Add(gbps);
        }

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "D\u2192H Transfer",
            DeviceName = profile.Name,
            MetricName = sizeLabel,
            Unit = "GB/s",
        };
        result.ComputeStats(measurements);
        return result;
    }

    private BenchmarkResult RunDeviceToDevice(
        Accelerator accel, DeviceProfile profile, BenchmarkOptions options,
        long sizeBytes, int floatCount, string sizeLabel)
    {
        var hostArray = new float[floatCount];
        for (int i = 0; i < floatCount; i++)
            hostArray[i] = i * 0.001f;

        using var src = accel.Allocate1D<float>(floatCount);
        using var dst = accel.Allocate1D<float>(floatCount);
        src.View.CopyFromCPU(hostArray);
        accel.Synchronize();

        // Warmup
        for (int w = 0; w < options.WarmupIterations; w++)
        {
            src.View.CopyTo(dst.View);
            accel.Synchronize();
        }

        var measurements = new List<double>();

        for (int i = 0; i < options.Iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            src.View.CopyTo(dst.View);
            accel.Synchronize();
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds;
            double gbps = sizeBytes / seconds / 1e9;
            measurements.Add(gbps);
        }

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "D\u2192D Copy",
            DeviceName = profile.Name,
            MetricName = sizeLabel,
            Unit = "GB/s",
        };
        result.ComputeStats(measurements);
        return result;
    }

    private void RenderBandwidthTable(
        string title,
        string benchmarkName,
        List<BenchmarkResult> results,
        IReadOnlyList<DeviceProfile> supportedProfiles,
        (long sizeBytes, string label)[] sizes)
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title($"[bold]{Markup.Escape(title)}[/]");

        table.AddColumn(new TableColumn("[bold]Size[/]"));
        foreach (var profile in supportedProfiles)
        {
            table.AddColumn(new TableColumn($"[{profile.Color} bold]{Markup.Escape(profile.Name)}[/]").RightAligned());
        }

        foreach (var (sizeBytes, label) in sizes)
        {
            var rowValues = new List<string> { $"[bold]{Markup.Escape(label)}[/]" };

            var deviceValues = new List<(double value, BenchmarkResult? result)>();

            foreach (var profile in supportedProfiles)
            {
                var result = results.FirstOrDefault(r =>
                    r.BenchmarkName == benchmarkName &&
                    r.DeviceName == profile.Name &&
                    r.MetricName == label);

                if (result == null)
                {
                    deviceValues.Add((0, null));
                }
                else
                {
                    deviceValues.Add((result.Best, result));
                }
            }

            // Find best value (highest bandwidth)
            double bestValue = 0;
            foreach (var dv in deviceValues)
            {
                if (dv.result != null && !dv.result.IsError && dv.value > bestValue)
                    bestValue = dv.value;
            }

            foreach (var dv in deviceValues)
            {
                if (dv.result == null)
                {
                    rowValues.Add("[dim]N/A[/]");
                }
                else if (dv.result.ErrorMessage == "skipped")
                {
                    rowValues.Add("[dim]skipped[/]");
                }
                else if (dv.result.IsError)
                {
                    rowValues.Add("[red]FAILED[/]");
                }
                else
                {
                    string valueStr = $"{dv.value:F2} GB/s";
                    if (Math.Abs(dv.value - bestValue) < 0.01 && deviceValues.Count(v => v.result != null && !v.result.IsError) > 1)
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
        AnsiConsole.WriteLine();
    }

    private static bool ShouldSkipSize(long sizeBytes, DeviceProfile profile)
    {
        if (profile.IsCpu) return false;
        long maxMemory = (long)(profile.MemorySize * 0.8);
        // For D->D we need 2x the buffer size, so check against that
        return sizeBytes * 2 > maxMemory;
    }

    private static void PrintInlineResult(string direction, BenchmarkResult result, DeviceProfile profile, string sizeLabel)
    {
        if (result.IsError)
        {
            AnsiConsole.MarkupLine($"    [red]{Markup.Escape(sizeLabel)} {direction}: FAILED[/]");
        }
        else
        {
            AnsiConsole.MarkupLine(
                $"    [{profile.Color}]{Markup.Escape(sizeLabel)} {direction}: {result.Best:F2} GB/s (avg {result.Average:F2}, stddev {result.StdDev:F2})[/]");
        }
    }

    private BenchmarkResult CreateErrorResult(string benchmarkName, string sizeLabel, DeviceProfile profile, Exception ex)
    {
        return new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = benchmarkName,
            DeviceName = profile.Name,
            MetricName = sizeLabel,
            Unit = "GB/s",
            ErrorMessage = ex.Message,
        };
    }

    private BenchmarkResult CreateSkippedResult(string benchmarkName, string sizeLabel, DeviceProfile profile)
    {
        return new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = benchmarkName,
            DeviceName = profile.Name,
            MetricName = sizeLabel,
            Unit = "GB/s",
            ErrorMessage = "skipped",
        };
    }
}
