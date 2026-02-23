using System.Diagnostics;
using GpuBench.Kernels;
using GpuBench.Models;
using ILGPU;
using ILGPU.Runtime;
using Spectre.Console;

namespace GpuBench.Benchmarks;

public sealed class MemoryPatterns : IBenchmarkSuite
{
    public string Name => "patterns";
    public string Description => "Memory access patterns";
    public bool SupportsDevice(DeviceProfile device) => true;

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

            // Test 1: Sequential vs Random Read
            try
            {
                var (seqResult, rndResult, ratioResult) = RunSequentialVsRandom(accel, profile, options);
                allResults.Add(seqResult);
                allResults.Add(rndResult);
                allResults.Add(ratioResult);
                PrintInlineResult(seqResult, profile);
                PrintInlineResult(rndResult, profile);
                PrintInlineResult(ratioResult, profile);
            }
            catch (Exception ex)
            {
                allResults.Add(CreateErrorResult("Sequential Read", "GB/s", profile, ex));
                allResults.Add(CreateErrorResult("Random Read", "GB/s", profile, ex));
                allResults.Add(CreateErrorResult("Seq/Rnd Ratio", "x", profile, ex));
                AnsiConsole.MarkupLine($"    [red]Sequential vs Random: FAILED - {Markup.Escape(ex.Message)}[/]");
            }

            // Test 2: Shared vs Global Memory
            try
            {
                var (globalResult, sharedResult, sharedRatioResult) = RunSharedVsGlobal(accel, profile, options);
                allResults.Add(globalResult);
                allResults.Add(sharedResult);
                allResults.Add(sharedRatioResult);
                PrintInlineResult(globalResult, profile);
                PrintInlineResult(sharedResult, profile);
                PrintInlineResult(sharedRatioResult, profile);
            }
            catch (Exception ex)
            {
                allResults.Add(CreateErrorResult("Global Repeated Read", "GB/s", profile, ex));
                allResults.Add(CreateErrorResult("Shared Repeated Read", "GB/s", profile, ex));
                allResults.Add(CreateErrorResult("Shared/Global Ratio", "x", profile, ex));
                AnsiConsole.MarkupLine($"    [red]Shared vs Global: FAILED - {Markup.Escape(ex.Message)}[/]");
            }

            // Test 3: Coalesced vs Strided Access
            try
            {
                var strideResults = RunStridedAccess(accel, profile, options);
                allResults.AddRange(strideResults);
                foreach (var r in strideResults)
                    PrintInlineResult(r, profile);
            }
            catch (Exception ex)
            {
                int[] strides = [1, 2, 4, 8, 16, 32];
                foreach (var s in strides)
                {
                    allResults.Add(CreateErrorResult($"Stride {s}", "GB/s", profile, ex));
                }
                AnsiConsole.MarkupLine($"    [red]Strided Access: FAILED - {Markup.Escape(ex.Message)}[/]");
            }

            AnsiConsole.WriteLine();
        }

        // Render summary tables
        RenderSequentialVsRandomTable(allResults, profiles);
        RenderSharedVsGlobalTable(allResults, profiles);
        RenderStridedAccessTable(allResults, profiles);

        return allResults;
    }

    private (BenchmarkResult seqResult, BenchmarkResult rndResult, BenchmarkResult ratioResult)
        RunSequentialVsRandom(Accelerator accel, DeviceProfile profile, BenchmarkOptions options)
    {
        int bufferSize = options.Quick ? 16_000_000 : 64_000_000;
        bufferSize = AdjustBufferSize(bufferSize, 3, profile); // input + indices + output

        var seqKernel = accel.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(MemoryKernels.SequentialReadKernel);

        var rndKernel = accel.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(MemoryKernels.RandomReadKernel);

        // Initialize input data with deterministic random
        var rng = new Random(42);
        var hostInput = new float[bufferSize];
        for (int i = 0; i < bufferSize; i++)
            hostInput[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        // Create shuffled index array (Fisher-Yates shuffle)
        var hostIndices = new int[bufferSize];
        for (int i = 0; i < bufferSize; i++)
            hostIndices[i] = i;
        var shuffleRng = new Random(42);
        for (int i = bufferSize - 1; i > 0; i--)
        {
            int j = shuffleRng.Next(i + 1);
            (hostIndices[i], hostIndices[j]) = (hostIndices[j], hostIndices[i]);
        }

        using var inputBuf = accel.Allocate1D<float>(bufferSize);
        using var outputBuf = accel.Allocate1D<float>(bufferSize);
        using var indicesBuf = accel.Allocate1D<int>(bufferSize);

        inputBuf.View.CopyFromCPU(hostInput);
        indicesBuf.View.CopyFromCPU(hostIndices);
        accel.Synchronize();

        // Measure sequential bandwidth
        var seqMeasurements = new List<double>();
        for (int w = 0; w < options.WarmupIterations; w++)
        {
            seqKernel(bufferSize, inputBuf.View, outputBuf.View);
            accel.Synchronize();
        }
        for (int i = 0; i < options.Iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            seqKernel(bufferSize, inputBuf.View, outputBuf.View);
            accel.Synchronize();
            sw.Stop();
            double bytes = (double)bufferSize * sizeof(float) * 2; // read + write
            double gbps = bytes / sw.Elapsed.TotalSeconds / 1e9;
            seqMeasurements.Add(gbps);
        }

        // Measure random bandwidth
        var rndMeasurements = new List<double>();
        for (int w = 0; w < options.WarmupIterations; w++)
        {
            rndKernel(bufferSize, inputBuf.View, indicesBuf.View, outputBuf.View);
            accel.Synchronize();
        }
        for (int i = 0; i < options.Iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            rndKernel(bufferSize, inputBuf.View, indicesBuf.View, outputBuf.View);
            accel.Synchronize();
            sw.Stop();
            double bytes = (double)bufferSize * sizeof(float) * 2; // read + write
            double gbps = bytes / sw.Elapsed.TotalSeconds / 1e9;
            rndMeasurements.Add(gbps);
        }

        string sizeLabel = $"{bufferSize / 1_000_000}M floats";

        var seqResult = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "Sequential Read",
            DeviceName = profile.Name,
            MetricName = sizeLabel,
            Unit = "GB/s",
        };
        seqResult.ComputeStats(seqMeasurements);

        var rndResult = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "Random Read",
            DeviceName = profile.Name,
            MetricName = sizeLabel,
            Unit = "GB/s",
        };
        rndResult.ComputeStats(rndMeasurements);

        double ratio = rndResult.Best > 0 ? seqResult.Best / rndResult.Best : 0;
        var ratioResult = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "Seq/Rnd Ratio",
            DeviceName = profile.Name,
            MetricName = sizeLabel,
            Unit = "x",
            Best = ratio,
            Average = ratio,
            Worst = ratio,
        };

        return (seqResult, rndResult, ratioResult);
    }

    private (BenchmarkResult globalResult, BenchmarkResult sharedResult, BenchmarkResult ratioResult)
        RunSharedVsGlobal(Accelerator accel, DeviceProfile profile, BenchmarkOptions options)
    {
        // Use a smaller buffer that aligns with group size for shared memory
        const int groupSize = 256;
        int bufferSize = options.Quick ? 1_000_000 : 4_000_000;
        // Round down to multiple of group size
        bufferSize = (bufferSize / groupSize) * groupSize;
        int repeats = 100;

        var globalKernel = accel.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(MemoryKernels.GlobalRepeatedReadKernel);

        // Initialize input data
        var rng = new Random(42);
        var hostInput = new float[bufferSize];
        for (int i = 0; i < bufferSize; i++)
            hostInput[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        using var inputBuf = accel.Allocate1D<float>(bufferSize);
        using var outputBuf = accel.Allocate1D<float>(bufferSize);
        inputBuf.View.CopyFromCPU(hostInput);
        accel.Synchronize();

        // Measure global memory repeated read
        var globalMeasurements = new List<double>();
        for (int w = 0; w < options.WarmupIterations; w++)
        {
            globalKernel(bufferSize, inputBuf.View, outputBuf.View, repeats);
            accel.Synchronize();
        }
        for (int i = 0; i < options.Iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            globalKernel(bufferSize, inputBuf.View, outputBuf.View, repeats);
            accel.Synchronize();
            sw.Stop();
            // Effective bandwidth: repeats reads + 1 write per element
            double bytes = (double)bufferSize * sizeof(float) * (repeats + 1);
            double gbps = bytes / sw.Elapsed.TotalSeconds / 1e9;
            globalMeasurements.Add(gbps);
        }

        // Measure shared memory repeated read
        var sharedMeasurements = new List<double>();
        bool sharedMemoryWorked = false;

        try
        {
            // Shared memory kernel needs explicit group size
            int numGroups = bufferSize / groupSize;
            var sharedKernel = accel.LoadStreamKernel<ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                int>(MemoryKernels.SharedRepeatedReadKernel);

            for (int w = 0; w < options.WarmupIterations; w++)
            {
                sharedKernel((numGroups, groupSize), inputBuf.View, outputBuf.View, repeats);
                accel.Synchronize();
            }
            for (int i = 0; i < options.Iterations; i++)
            {
                var sw = Stopwatch.StartNew();
                sharedKernel((numGroups, groupSize), inputBuf.View, outputBuf.View, repeats);
                accel.Synchronize();
                sw.Stop();
                double bytes = (double)bufferSize * sizeof(float) * (repeats + 1);
                double gbps = bytes / sw.Elapsed.TotalSeconds / 1e9;
                sharedMeasurements.Add(gbps);
            }
            sharedMemoryWorked = true;
        }
        catch (Exception ex)
        {
            // Shared memory may not work on CPU accelerator or certain configurations.
            // Fall back gracefully.
            AnsiConsole.MarkupLine($"    [yellow]Shared memory test skipped: {Markup.Escape(ex.Message)}[/]");
        }

        string sizeLabel = $"{bufferSize / 1_000_000}M floats";

        var globalResult = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "Global Repeated Read",
            DeviceName = profile.Name,
            MetricName = $"{sizeLabel} x{repeats}",
            Unit = "GB/s",
        };
        globalResult.ComputeStats(globalMeasurements);

        BenchmarkResult sharedResult;
        BenchmarkResult ratioResult;

        if (sharedMemoryWorked && sharedMeasurements.Count > 0)
        {
            sharedResult = new BenchmarkResult
            {
                SuiteName = Name,
                BenchmarkName = "Shared Repeated Read",
                DeviceName = profile.Name,
                MetricName = $"{sizeLabel} x{repeats}",
                Unit = "GB/s",
            };
            sharedResult.ComputeStats(sharedMeasurements);

            double ratio = globalResult.Best > 0 ? sharedResult.Best / globalResult.Best : 0;
            ratioResult = new BenchmarkResult
            {
                SuiteName = Name,
                BenchmarkName = "Shared/Global Ratio",
                DeviceName = profile.Name,
                MetricName = $"{sizeLabel} x{repeats}",
                Unit = "x",
                Best = ratio,
                Average = ratio,
                Worst = ratio,
            };
        }
        else
        {
            sharedResult = new BenchmarkResult
            {
                SuiteName = Name,
                BenchmarkName = "Shared Repeated Read",
                DeviceName = profile.Name,
                MetricName = $"{sizeLabel} x{repeats}",
                Unit = "GB/s",
                ErrorMessage = "shared memory not supported on this device",
            };
            ratioResult = new BenchmarkResult
            {
                SuiteName = Name,
                BenchmarkName = "Shared/Global Ratio",
                DeviceName = profile.Name,
                MetricName = $"{sizeLabel} x{repeats}",
                Unit = "x",
                ErrorMessage = "shared memory not supported on this device",
            };
        }

        return (globalResult, sharedResult, ratioResult);
    }

    private List<BenchmarkResult> RunStridedAccess(
        Accelerator accel, DeviceProfile profile, BenchmarkOptions options)
    {
        int bufferSize = options.Quick ? 16_000_000 : 64_000_000;
        bufferSize = AdjustBufferSize(bufferSize, 2, profile); // input + output

        int[] strides = [1, 2, 4, 8, 16, 32];

        var stridedKernel = accel.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int>(MemoryKernels.StridedReadKernel);

        // Initialize input data
        var rng = new Random(42);
        var hostInput = new float[bufferSize];
        for (int i = 0; i < bufferSize; i++)
            hostInput[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        using var inputBuf = accel.Allocate1D<float>(bufferSize);
        using var outputBuf = accel.Allocate1D<float>(bufferSize);
        inputBuf.View.CopyFromCPU(hostInput);
        accel.Synchronize();

        var results = new List<BenchmarkResult>();

        foreach (int stride in strides)
        {
            var measurements = new List<double>();

            for (int w = 0; w < options.WarmupIterations; w++)
            {
                stridedKernel(bufferSize, inputBuf.View, outputBuf.View, stride, bufferSize);
                accel.Synchronize();
            }

            for (int i = 0; i < options.Iterations; i++)
            {
                var sw = Stopwatch.StartNew();
                stridedKernel(bufferSize, inputBuf.View, outputBuf.View, stride, bufferSize);
                accel.Synchronize();
                sw.Stop();
                double bytes = (double)bufferSize * sizeof(float) * 2; // read + write
                double gbps = bytes / sw.Elapsed.TotalSeconds / 1e9;
                measurements.Add(gbps);
            }

            var result = new BenchmarkResult
            {
                SuiteName = Name,
                BenchmarkName = $"Stride {stride}",
                DeviceName = profile.Name,
                MetricName = $"{bufferSize / 1_000_000}M floats",
                Unit = "GB/s",
            };
            result.ComputeStats(measurements);
            results.Add(result);
        }

        return results;
    }

    private void RenderSequentialVsRandomTable(List<BenchmarkResult> results, IReadOnlyList<DeviceProfile> profiles)
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]Sequential vs Random Access[/]");

        table.AddColumn(new TableColumn("[bold]Pattern[/]"));
        foreach (var profile in profiles)
            table.AddColumn(new TableColumn($"[{profile.Color} bold]{Markup.Escape(profile.Name)}[/]").RightAligned());

        foreach (var benchName in new[] { "Sequential Read", "Random Read", "Seq/Rnd Ratio" })
        {
            var rowValues = new List<string> { $"[bold]{Markup.Escape(benchName)}[/]" };
            string unit = benchName.Contains("Ratio") ? "x" : "GB/s";

            foreach (var profile in profiles)
            {
                var result = results.FirstOrDefault(r =>
                    r.BenchmarkName == benchName && r.DeviceName == profile.Name);

                if (result == null)
                    rowValues.Add("[dim]N/A[/]");
                else if (result.IsError)
                    rowValues.Add("[red]FAILED[/]");
                else
                    rowValues.Add($"{result.Best:F2} {unit}");
            }

            table.AddRow(rowValues.ToArray());
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    private void RenderSharedVsGlobalTable(List<BenchmarkResult> results, IReadOnlyList<DeviceProfile> profiles)
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]Shared vs Global Memory[/]");

        table.AddColumn(new TableColumn("[bold]Pattern[/]"));
        foreach (var profile in profiles)
            table.AddColumn(new TableColumn($"[{profile.Color} bold]{Markup.Escape(profile.Name)}[/]").RightAligned());

        foreach (var benchName in new[] { "Global Repeated Read", "Shared Repeated Read", "Shared/Global Ratio" })
        {
            var rowValues = new List<string> { $"[bold]{Markup.Escape(benchName)}[/]" };
            string unit = benchName.Contains("Ratio") ? "x" : "GB/s";

            foreach (var profile in profiles)
            {
                var result = results.FirstOrDefault(r =>
                    r.BenchmarkName == benchName && r.DeviceName == profile.Name);

                if (result == null)
                    rowValues.Add("[dim]N/A[/]");
                else if (result.IsError)
                    rowValues.Add($"[dim]{Markup.Escape(result.ErrorMessage ?? "N/A")}[/]");
                else
                    rowValues.Add($"{result.Best:F2} {unit}");
            }

            table.AddRow(rowValues.ToArray());
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    private void RenderStridedAccessTable(List<BenchmarkResult> results, IReadOnlyList<DeviceProfile> profiles)
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]Coalesced vs Strided Access[/]");

        table.AddColumn(new TableColumn("[bold]Stride[/]"));
        foreach (var profile in profiles)
            table.AddColumn(new TableColumn($"[{profile.Color} bold]{Markup.Escape(profile.Name)}[/]").RightAligned());

        int[] strides = [1, 2, 4, 8, 16, 32];

        foreach (int stride in strides)
        {
            string benchName = $"Stride {stride}";
            var rowValues = new List<string> { $"[bold]{benchName}[/]" };

            var deviceValues = new List<(double value, BenchmarkResult? result)>();

            foreach (var profile in profiles)
            {
                var result = results.FirstOrDefault(r =>
                    r.BenchmarkName == benchName && r.DeviceName == profile.Name);

                if (result == null)
                    deviceValues.Add((0, null));
                else if (result.IsError)
                    deviceValues.Add((0, result));
                else
                    deviceValues.Add((result.Best, result));
            }

            // Find best (highest bandwidth)
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
                else if (dv.result.IsError)
                {
                    rowValues.Add("[red]FAILED[/]");
                }
                else
                {
                    string valueStr = $"{dv.value:F2} GB/s";
                    if (Math.Abs(dv.value - bestValue) < 0.01 && deviceValues.Count(v => v.result != null && !v.result.IsError) > 1)
                        rowValues.Add($"[bold]{valueStr}[/]");
                    else
                        rowValues.Add(valueStr);
                }
            }

            table.AddRow(rowValues.ToArray());
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
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
                $"    [{profile.Color}]{Markup.Escape(result.BenchmarkName)}: {result.Best:F2} {result.Unit} (avg {result.Average:F2}, stddev {result.StdDev:F2})[/]");
        }
    }

    private BenchmarkResult CreateErrorResult(string benchmarkName, string unit, DeviceProfile profile, Exception ex)
    {
        return new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = benchmarkName,
            DeviceName = profile.Name,
            MetricName = "error",
            Unit = unit,
            ErrorMessage = ex.Message,
        };
    }

    private static int AdjustBufferSize(int bufferSize, int numBuffers, DeviceProfile profile)
    {
        if (profile.IsCpu) return bufferSize;

        long maxMemory = (long)(profile.MemorySize * 0.8);
        long required = (long)bufferSize * sizeof(float) * numBuffers;

        while (required > maxMemory && bufferSize > 1_000_000)
        {
            bufferSize /= 2;
            required = (long)bufferSize * sizeof(float) * numBuffers;
        }

        return bufferSize;
    }
}
