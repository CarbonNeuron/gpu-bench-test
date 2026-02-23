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

    public List<BenchmarkResult> Run(
        IReadOnlyList<Accelerator> accelerators,
        IReadOnlyList<DeviceProfile> profiles,
        BenchmarkOptions options)
    {
        var allResults = new List<BenchmarkResult>();

        // Store vector results for cross-device verification
        var vectorAddResults = new Dictionary<int, float[]>();
        var vectorFmaResults = new Dictionary<int, float[]>();
        int referenceDeviceIdx = -1;

        for (int di = 0; di < accelerators.Count; di++)
        {
            var accel = accelerators[di];
            var profile = profiles[di];

            if (!SupportsDevice(profile)) continue;

            AnsiConsole.MarkupLine($"  [{profile.Color} bold]{Markup.Escape(profile.Name)}[/]");

            // FP32 FMA
            try
            {
                var result = RunFp32Throughput(accel, profile, options);
                allResults.Add(result);
                PrintInlineResult(result, profile);
            }
            catch (Exception ex)
            {
                var result = CreateErrorResult("FP32 Throughput", "GFLOPS", profile, ex);
                allResults.Add(result);
                AnsiConsole.MarkupLine($"    [red]FP32 Throughput: FAILED - {Markup.Escape(ex.Message)}[/]");
            }

            // FP64 FMA
            try
            {
                var result = RunFp64Throughput(accel, profile, options);
                allResults.Add(result);
                PrintInlineResult(result, profile);
            }
            catch (Exception ex)
            {
                var result = CreateErrorResult("FP64 Throughput", "GFLOPS", profile, ex);
                allResults.Add(result);
                AnsiConsole.MarkupLine($"    [red]FP64 Throughput: FAILED - {Markup.Escape(ex.Message)}[/]");
            }

            // Int32 FMA
            try
            {
                var result = RunInt32Throughput(accel, profile, options);
                allResults.Add(result);
                PrintInlineResult(result, profile);
            }
            catch (Exception ex)
            {
                var result = CreateErrorResult("Int32 Throughput", "GOPS", profile, ex);
                allResults.Add(result);
                AnsiConsole.MarkupLine($"    [red]Int32 Throughput: FAILED - {Markup.Escape(ex.Message)}[/]");
            }

            // Vector Add
            try
            {
                var (result, outputData) = RunVectorAdd(accel, profile, options);
                allResults.Add(result);
                PrintInlineResult(result, profile);

                if (!result.IsError && outputData != null)
                {
                    vectorAddResults[di] = outputData;
                    if (referenceDeviceIdx < 0) referenceDeviceIdx = di;
                }
            }
            catch (Exception ex)
            {
                var result = CreateErrorResult("Vector Add", "GB/s", profile, ex);
                allResults.Add(result);
                AnsiConsole.MarkupLine($"    [red]Vector Add: FAILED - {Markup.Escape(ex.Message)}[/]");
            }

            // Vector FMA
            try
            {
                var (result, outputData) = RunVectorFma(accel, profile, options);
                allResults.Add(result);
                PrintInlineResult(result, profile);

                if (!result.IsError && outputData != null)
                {
                    vectorFmaResults[di] = outputData;
                }
            }
            catch (Exception ex)
            {
                var result = CreateErrorResult("Vector FMA", "GFLOPS", profile, ex);
                allResults.Add(result);
                AnsiConsole.MarkupLine($"    [red]Vector FMA: FAILED - {Markup.Escape(ex.Message)}[/]");
            }

            AnsiConsole.WriteLine();
        }

        // Cross-device verification for vector results
        VerifyCrossDevice(allResults, vectorAddResults, "Vector Add", referenceDeviceIdx);
        VerifyCrossDevice(allResults, vectorFmaResults, "Vector FMA", referenceDeviceIdx);

        // Render summary table
        RenderSummaryTable(allResults, profiles);

        return allResults;
    }

    private BenchmarkResult RunFp32Throughput(Accelerator accel, DeviceProfile profile, BenchmarkOptions options)
    {
        var threadCounts = GetThreadCounts(options);
        int threadCount = SelectThreadCount(threadCounts, sizeof(float), profile);

        var kernel = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(
            ComputeKernels.Fp32FmaKernel);

        using var buffer = accel.Allocate1D<float>(threadCount);
        var measurements = new List<double>();

        // Warmup
        for (int w = 0; w < options.WarmupIterations; w++)
        {
            kernel(threadCount, buffer.View);
            accel.Synchronize();
        }

        // Timed iterations
        for (int i = 0; i < options.Iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            kernel(threadCount, buffer.View);
            accel.Synchronize();
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds;
            double gflops = (double)threadCount * ComputeKernels.FmaOpsPerThread * ComputeKernels.FlopsPerFma / seconds / 1e9;
            measurements.Add(gflops);
        }

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "FP32 Throughput",
            DeviceName = profile.Name,
            MetricName = $"{threadCount / 1_000_000}M threads",
            Unit = "GFLOPS",
        };
        result.ComputeStats(measurements);
        return result;
    }

    private BenchmarkResult RunFp64Throughput(Accelerator accel, DeviceProfile profile, BenchmarkOptions options)
    {
        var threadCounts = GetThreadCounts(options);
        int threadCount = SelectThreadCount(threadCounts, sizeof(double), profile);

        var kernel = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<double, Stride1D.Dense>>(
            ComputeKernels.Fp64FmaKernel);

        using var buffer = accel.Allocate1D<double>(threadCount);
        var measurements = new List<double>();

        for (int w = 0; w < options.WarmupIterations; w++)
        {
            kernel(threadCount, buffer.View);
            accel.Synchronize();
        }

        for (int i = 0; i < options.Iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            kernel(threadCount, buffer.View);
            accel.Synchronize();
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds;
            double gflops = (double)threadCount * ComputeKernels.FmaOpsPerThread * ComputeKernels.FlopsPerFma / seconds / 1e9;
            measurements.Add(gflops);
        }

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "FP64 Throughput",
            DeviceName = profile.Name,
            MetricName = $"{threadCount / 1_000_000}M threads",
            Unit = "GFLOPS",
        };
        result.ComputeStats(measurements);
        return result;
    }

    private BenchmarkResult RunInt32Throughput(Accelerator accel, DeviceProfile profile, BenchmarkOptions options)
    {
        var threadCounts = GetThreadCounts(options);
        int threadCount = SelectThreadCount(threadCounts, sizeof(int), profile);

        var kernel = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>>(
            ComputeKernels.Int32FmaKernel);

        using var buffer = accel.Allocate1D<int>(threadCount);
        var measurements = new List<double>();

        for (int w = 0; w < options.WarmupIterations; w++)
        {
            kernel(threadCount, buffer.View);
            accel.Synchronize();
        }

        for (int i = 0; i < options.Iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            kernel(threadCount, buffer.View);
            accel.Synchronize();
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds;
            // Int32: 2 ops per FMA iteration (1 mul + 1 add)
            double gops = (double)threadCount * ComputeKernels.FmaOpsPerThread * 2.0 / seconds / 1e9;
            measurements.Add(gops);
        }

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "Int32 Throughput",
            DeviceName = profile.Name,
            MetricName = $"{threadCount / 1_000_000}M threads",
            Unit = "GOPS",
        };
        result.ComputeStats(measurements);
        return result;
    }

    private (BenchmarkResult result, float[]? outputData) RunVectorAdd(
        Accelerator accel, DeviceProfile profile, BenchmarkOptions options)
    {
        int vectorSize = GetVectorSize(options);
        vectorSize = AdjustVectorSizeForMemory(vectorSize, 3, profile);

        var kernel = accel.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(ComputeKernels.VectorAddKernel);

        // Initialize host arrays with deterministic random
        var rng = new Random(42);
        var hostA = new float[vectorSize];
        var hostB = new float[vectorSize];
        for (int i = 0; i < vectorSize; i++)
        {
            hostA[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            hostB[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        using var bufA = accel.Allocate1D<float>(vectorSize);
        using var bufB = accel.Allocate1D<float>(vectorSize);
        using var bufC = accel.Allocate1D<float>(vectorSize);

        bufA.View.CopyFromCPU(hostA);
        bufB.View.CopyFromCPU(hostB);

        var measurements = new List<double>();

        // Warmup
        for (int w = 0; w < options.WarmupIterations; w++)
        {
            kernel(vectorSize, bufA.View, bufB.View, bufC.View);
            accel.Synchronize();
        }

        // Timed iterations
        for (int i = 0; i < options.Iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            kernel(vectorSize, bufA.View, bufB.View, bufC.View);
            accel.Synchronize();
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds;
            // 3 memory operations: read A, read B, write C
            double gbps = (double)vectorSize * sizeof(float) * 3.0 / seconds / 1e9;
            measurements.Add(gbps);
        }

        // Get results back for verification
        var outputData = bufC.GetAsArray1D();

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "Vector Add",
            DeviceName = profile.Name,
            MetricName = $"{vectorSize / 1_000_000}M elements",
            Unit = "GB/s",
        };
        result.ComputeStats(measurements);

        return (result, outputData);
    }

    private (BenchmarkResult result, float[]? outputData) RunVectorFma(
        Accelerator accel, DeviceProfile profile, BenchmarkOptions options)
    {
        int vectorSize = GetVectorSize(options);
        vectorSize = AdjustVectorSizeForMemory(vectorSize, 3, profile);

        var kernel = accel.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(ComputeKernels.VectorFmaKernel);

        // Initialize host arrays with deterministic random
        var rng = new Random(42);
        var hostA = new float[vectorSize];
        var hostB = new float[vectorSize];
        var hostC = new float[vectorSize];
        for (int i = 0; i < vectorSize; i++)
        {
            hostA[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            hostB[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            hostC[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        using var bufA = accel.Allocate1D<float>(vectorSize);
        using var bufB = accel.Allocate1D<float>(vectorSize);
        using var bufC = accel.Allocate1D<float>(vectorSize);

        bufA.View.CopyFromCPU(hostA);
        bufB.View.CopyFromCPU(hostB);
        bufC.View.CopyFromCPU(hostC);

        var measurements = new List<double>();

        // Warmup
        for (int w = 0; w < options.WarmupIterations; w++)
        {
            kernel(vectorSize, bufA.View, bufB.View, bufC.View);
            accel.Synchronize();
        }

        // Timed iterations
        for (int i = 0; i < options.Iterations; i++)
        {
            // Re-copy C before each iteration since FMA modifies it in place
            bufC.View.CopyFromCPU(hostC);
            accel.Synchronize();

            var sw = Stopwatch.StartNew();
            kernel(vectorSize, bufA.View, bufB.View, bufC.View);
            accel.Synchronize();
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds;
            // 2 flops per element: 1 mul + 1 add
            double gflops = (double)vectorSize * 2.0 / seconds / 1e9;
            measurements.Add(gflops);
        }

        // Get results back for verification (run one final clean pass)
        bufC.View.CopyFromCPU(hostC);
        accel.Synchronize();
        kernel(vectorSize, bufA.View, bufB.View, bufC.View);
        accel.Synchronize();
        var outputData = bufC.GetAsArray1D();

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = "Vector FMA",
            DeviceName = profile.Name,
            MetricName = $"{vectorSize / 1_000_000}M elements",
            Unit = "GFLOPS",
        };
        result.ComputeStats(measurements);

        return (result, outputData);
    }

    private static void VerifyCrossDevice(
        List<BenchmarkResult> allResults,
        Dictionary<int, float[]> deviceResults,
        string benchmarkName,
        int referenceDeviceIdx)
    {
        if (deviceResults.Count < 1) return;

        // Mark reference device
        foreach (var result in allResults)
        {
            if (result.BenchmarkName != benchmarkName) continue;
            if (result.IsError) continue;

            // Find the device index for this result
            int deviceIdx = -1;
            foreach (var kvp in deviceResults)
            {
                // Match by result iteration order
                if (deviceIdx < 0) deviceIdx = kvp.Key;
            }
        }

        // Find reference data
        if (!deviceResults.TryGetValue(referenceDeviceIdx, out var refData)) return;

        foreach (var kvp in deviceResults)
        {
            var matchingResults = allResults.Where(r =>
                r.BenchmarkName == benchmarkName && !r.IsError).ToList();

            // Results are in device order, so index matches
            int resultIndex = 0;
            foreach (var result in matchingResults)
            {
                int currentDeviceIdx = deviceResults.Keys.ElementAt(resultIndex);
                if (currentDeviceIdx == referenceDeviceIdx)
                {
                    result.Verification = VerificationStatus.Reference;
                }
                else
                {
                    var testData = deviceResults[currentDeviceIdx];
                    bool passed = VerifyArrays(refData, testData, 1e-2f);
                    result.Verification = passed ? VerificationStatus.Passed : VerificationStatus.Failed;
                }
                resultIndex++;
                if (resultIndex >= deviceResults.Count) break;
            }
            break; // Only need one pass
        }
    }

    private static bool VerifyArrays(float[] reference, float[] test, float epsilon)
    {
        int length = Math.Min(reference.Length, test.Length);
        if (reference.Length != test.Length) return false;

        for (int i = 0; i < length; i += 1000)
        {
            if (MathF.Abs(reference[i] - test[i]) > epsilon)
                return false;
        }
        return true;
    }

    private void RenderSummaryTable(List<BenchmarkResult> results, IReadOnlyList<DeviceProfile> profiles)
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]Compute Throughput[/]");

        table.AddColumn(new TableColumn("[bold]Benchmark[/]"));
        foreach (var profile in profiles)
        {
            table.AddColumn(new TableColumn($"[{profile.Color} bold]{Markup.Escape(profile.Name)}[/]").RightAligned());
        }

        var benchmarkNames = new[] { "FP32 Throughput", "FP64 Throughput", "Int32 Throughput", "Vector Add", "Vector FMA" };

        foreach (var benchName in benchmarkNames)
        {
            var rowValues = new List<string>();
            rowValues.Add($"[bold]{Markup.Escape(benchName)}[/]");

            var deviceValues = new List<(int idx, double value, string unit, BenchmarkResult result)>();

            foreach (var profile in profiles)
            {
                var result = results.FirstOrDefault(r =>
                    r.BenchmarkName == benchName && r.DeviceName == profile.Name);

                if (result == null)
                {
                    deviceValues.Add((-1, 0, "", null!));
                }
                else if (result.IsError)
                {
                    deviceValues.Add((-1, 0, result.Unit, result));
                }
                else
                {
                    deviceValues.Add((deviceValues.Count, result.Best, result.Unit, result));
                }
            }

            // Find best value
            double bestValue = 0;
            foreach (var dv in deviceValues)
            {
                if (dv.idx >= 0 && dv.value > bestValue)
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
                    rowValues.Add($"[red]FAILED[/]");
                }
                else
                {
                    string verificationMark = dv.result.Verification switch
                    {
                        VerificationStatus.Passed => " [green]\u2713[/]",
                        VerificationStatus.Failed => " [red]\u26A0[/]",
                        VerificationStatus.Reference => " [dim]ref[/]",
                        _ => ""
                    };

                    string valueStr = $"{dv.value:F1} {dv.result.Unit}";
                    if (Math.Abs(dv.value - bestValue) < 0.01 && deviceValues.Count(v => v.idx >= 0) > 1)
                    {
                        rowValues.Add($"[bold]{valueStr}[/]{verificationMark}");
                    }
                    else
                    {
                        rowValues.Add($"{valueStr}{verificationMark}");
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
                $"    [{profile.Color}]{Markup.Escape(result.BenchmarkName)}: {result.Best:F1} {result.Unit} (avg {result.Average:F1}, stddev {result.StdDev:F1})[/]");
        }
    }

    private static BenchmarkResult CreateErrorResult(string benchmarkName, string unit, DeviceProfile profile, Exception ex)
    {
        return new BenchmarkResult
        {
            SuiteName = "compute",
            BenchmarkName = benchmarkName,
            DeviceName = profile.Name,
            MetricName = "error",
            Unit = unit,
            ErrorMessage = ex.Message,
        };
    }

    private static int[] GetThreadCounts(BenchmarkOptions options)
    {
        return options.Quick
            ? [1_000_000, 4_000_000]
            : [1_000_000, 4_000_000, 16_000_000];
    }

    private static int SelectThreadCount(int[] threadCounts, int elementSize, DeviceProfile profile)
    {
        // Use the largest thread count that fits in 80% of device memory
        // Skip memory check for CPU
        if (profile.IsCpu)
            return threadCounts[^1];

        long maxMemory = (long)(profile.MemorySize * 0.8);
        int selected = threadCounts[0];

        foreach (var count in threadCounts)
        {
            long requiredMemory = (long)count * elementSize;
            if (requiredMemory <= maxMemory)
                selected = count;
        }

        return selected;
    }

    private static int GetVectorSize(BenchmarkOptions options)
    {
        return options.Quick ? 16_000_000 : 64_000_000;
    }

    private static int AdjustVectorSizeForMemory(int vectorSize, int numBuffers, DeviceProfile profile)
    {
        if (profile.IsCpu) return vectorSize;

        long maxMemory = (long)(profile.MemorySize * 0.8);
        long required = (long)vectorSize * sizeof(float) * numBuffers;

        while (required > maxMemory && vectorSize > 1_000_000)
        {
            vectorSize /= 2;
            required = (long)vectorSize * sizeof(float) * numBuffers;
        }

        return vectorSize;
    }
}
