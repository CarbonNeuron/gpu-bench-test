using System.Diagnostics;
using GpuBench.Kernels;
using GpuBench.Models;
using ILGPU;
using ILGPU.Runtime;
using Spectre.Console;

namespace GpuBench.Benchmarks;

public sealed class MatrixMultiply : IBenchmarkSuite
{
    public string Name => "matmul";
    public string Description => "Matrix multiplication (naive + tiled)";
    public bool SupportsDevice(DeviceProfile device) => true;

    public List<BenchmarkResult> Run(
        IReadOnlyList<Accelerator> accelerators,
        IReadOnlyList<DeviceProfile> profiles,
        BenchmarkOptions options)
    {
        var allResults = new List<BenchmarkResult>();
        var sizes = GetSizes(options);

        // Store reference results for cross-device verification (keyed by size)
        var referenceResults = new Dictionary<int, float[]>();

        foreach (int n in sizes)
        {
            AnsiConsole.MarkupLine($"  [bold]Matrix size: {n}x{n}[/] ({(long)n * n * sizeof(float) * 3 / (1024 * 1024)} MB for 3 matrices)");

            // Track per-device naive times for speedup calculation
            var naiveTimesPerDevice = new Dictionary<int, double>();
            var tiledTimesPerDevice = new Dictionary<int, double>();

            for (int di = 0; di < accelerators.Count; di++)
            {
                var accel = accelerators[di];
                var profile = profiles[di];

                if (!SupportsDevice(profile)) continue;

                // Memory safety check
                long requiredMemory = (long)n * n * sizeof(float) * 3;
                long maxMemory = profile.IsCpu ? long.MaxValue : (long)(profile.MemorySize * 0.8);
                if (requiredMemory > maxMemory)
                {
                    AnsiConsole.MarkupLine($"    [{profile.Color}]{Markup.Escape(profile.Name)}: [dim]Skipped (requires {requiredMemory / (1024 * 1024)} MB, device has {profile.MemorySize / (1024 * 1024)} MB)[/][/]");
                    continue;
                }

                // CPU limited to sizes <= 2048 unless --full
                if (profile.IsCpu && n > 2048 && !options.Full)
                {
                    AnsiConsole.MarkupLine($"    [{profile.Color}]{Markup.Escape(profile.Name)}: [dim]Skipped for CPU (size > 2048, use --full to include)[/][/]");
                    continue;
                }

                AnsiConsole.MarkupLine($"    [{profile.Color} bold]{Markup.Escape(profile.Name)}[/]");

                // Initialize host matrices with deterministic random
                var rng = new Random(42);
                var hostA = new float[n * n];
                var hostB = new float[n * n];
                for (int i = 0; i < n * n; i++)
                {
                    hostA[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                    hostB[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                }

                using var bufA = accel.Allocate1D<float>(n * n);
                using var bufB = accel.Allocate1D<float>(n * n);
                using var bufC = accel.Allocate1D<float>(n * n);

                bufA.View.CopyFromCPU(hostA);
                bufB.View.CopyFromCPU(hostB);

                // --- Naive kernel ---
                try
                {
                    var naiveResult = RunNaiveKernel(accel, profile, options, bufA, bufB, bufC, n);
                    allResults.Add(naiveResult);
                    PrintInlineResult(naiveResult, profile);
                    if (!naiveResult.IsError)
                        naiveTimesPerDevice[di] = naiveResult.Best;
                }
                catch (Exception ex)
                {
                    var result = CreateErrorResult($"Naive {n}x{n}", "GFLOPS", profile, ex);
                    allResults.Add(result);
                    AnsiConsole.MarkupLine($"      [red]Naive {n}x{n}: FAILED - {Markup.Escape(ex.Message)}[/]");
                }

                // --- Tiled kernel ---
                float[]? tiledOutput = null;
                int requiredGroupSize = MatMulKernels.TileSize * MatMulKernels.TileSize;
                if (profile.MaxThreadsPerGroup < requiredGroupSize)
                {
                    AnsiConsole.MarkupLine(
                        $"      [{profile.Color}]Tiled {n}x{n}: [dim]Skipped (device max threads/group = {profile.MaxThreadsPerGroup}, need {requiredGroupSize})[/][/]");
                    var skipResult = new BenchmarkResult
                    {
                        SuiteName = Name,
                        BenchmarkName = $"Tiled {n}x{n}",
                        DeviceName = profile.Name,
                        MetricName = $"{n}x{n}",
                        Unit = "GFLOPS",
                        ErrorMessage = $"Skipped: device max threads/group ({profile.MaxThreadsPerGroup}) < required ({requiredGroupSize})",
                    };
                    allResults.Add(skipResult);
                }
                else
                try
                {
                    var (tiledResult, outputData) = RunTiledKernel(accel, profile, options, bufA, bufB, bufC, n);
                    allResults.Add(tiledResult);
                    tiledOutput = outputData;

                    if (!tiledResult.IsError)
                    {
                        tiledTimesPerDevice[di] = tiledResult.Best;

                        // Show speedup over naive
                        if (naiveTimesPerDevice.TryGetValue(di, out double naiveGflops) && naiveGflops > 0)
                        {
                            double speedup = tiledResult.Best / naiveGflops;
                            AnsiConsole.MarkupLine(
                                $"      [{profile.Color}]Tiled {n}x{n}: {tiledResult.Best:F1} {tiledResult.Unit} (avg {tiledResult.Average:F1}, stddev {tiledResult.StdDev:F1}) [bold]{speedup:F2}x[/] vs naive[/]");
                        }
                        else
                        {
                            PrintInlineResult(tiledResult, profile);
                        }
                    }
                    else
                    {
                        PrintInlineResult(tiledResult, profile);
                    }
                }
                catch (Exception ex)
                {
                    var result = CreateErrorResult($"Tiled {n}x{n}", "GFLOPS", profile, ex);
                    allResults.Add(result);
                    AnsiConsole.MarkupLine($"      [red]Tiled {n}x{n}: FAILED - {Markup.Escape(ex.Message)}[/]");
                }

                // Cross-device verification on tiled results
                if (tiledOutput != null)
                {
                    if (!referenceResults.ContainsKey(n))
                    {
                        // First device is the reference
                        referenceResults[n] = tiledOutput;

                        // Mark tiled result as reference
                        var tiledRef = allResults.LastOrDefault(r =>
                            r.BenchmarkName == $"Tiled {n}x{n}" && r.DeviceName == profile.Name && !r.IsError);
                        if (tiledRef != null)
                            tiledRef.Verification = VerificationStatus.Reference;
                    }
                    else
                    {
                        // Verify against reference
                        bool passed = VerifyArrays(referenceResults[n], tiledOutput, 1e-2f, n * n);
                        var tiledVer = allResults.LastOrDefault(r =>
                            r.BenchmarkName == $"Tiled {n}x{n}" && r.DeviceName == profile.Name && !r.IsError);
                        if (tiledVer != null)
                        {
                            tiledVer.Verification = passed ? VerificationStatus.Passed : VerificationStatus.Failed;
                            string status = passed ? "[green]PASS[/]" : "[red]FAIL[/]";
                            AnsiConsole.MarkupLine($"      Verification: {status}");
                        }
                    }
                }
            }

            AnsiConsole.WriteLine();
        }

        // Render summary table
        RenderSummaryTable(allResults, profiles, sizes);

        return allResults;
    }

    private BenchmarkResult RunNaiveKernel(
        Accelerator accel,
        DeviceProfile profile,
        BenchmarkOptions options,
        MemoryBuffer1D<float, Stride1D.Dense> bufA,
        MemoryBuffer1D<float, Stride1D.Dense> bufB,
        MemoryBuffer1D<float, Stride1D.Dense> bufC,
        int n)
    {
        var kernel = accel.LoadAutoGroupedStreamKernel<Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(MatMulKernels.NaiveMatMulKernel);

        var measurements = new List<double>();
        var extent = new Index2D(n, n);

        // Warmup
        for (int w = 0; w < options.WarmupIterations; w++)
        {
            kernel(extent, bufA.View, bufB.View, bufC.View, n);
            accel.Synchronize();
        }

        // Timed iterations
        for (int i = 0; i < options.Iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            kernel(extent, bufA.View, bufB.View, bufC.View, n);
            accel.Synchronize();
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds;
            double gflops = 2.0 * n * n * n / seconds / 1e9;
            measurements.Add(gflops);
        }

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = $"Naive {n}x{n}",
            DeviceName = profile.Name,
            MetricName = $"{n}x{n}",
            Unit = "GFLOPS",
        };
        result.ComputeStats(measurements);
        return result;
    }

    private (BenchmarkResult result, float[]? outputData) RunTiledKernel(
        Accelerator accel,
        DeviceProfile profile,
        BenchmarkOptions options,
        MemoryBuffer1D<float, Stride1D.Dense> bufA,
        MemoryBuffer1D<float, Stride1D.Dense> bufB,
        MemoryBuffer1D<float, Stride1D.Dense> bufC,
        int n)
    {
        var kernel = accel.LoadStreamKernel<
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(MatMulKernels.TiledMatMulKernel);

        int tileSize = MatMulKernels.TileSize;
        int gridDimX = (n + tileSize - 1) / tileSize;
        int gridDimY = (n + tileSize - 1) / tileSize;

        var gridDim = new Index2D(gridDimX, gridDimY);
        var groupDim = new Index2D(tileSize, tileSize);

        var measurements = new List<double>();

        // Warmup
        for (int w = 0; w < options.WarmupIterations; w++)
        {
            kernel((gridDim, groupDim), bufA.View, bufB.View, bufC.View, n);
            accel.Synchronize();
        }

        // Timed iterations
        for (int i = 0; i < options.Iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            kernel((gridDim, groupDim), bufA.View, bufB.View, bufC.View, n);
            accel.Synchronize();
            sw.Stop();

            double seconds = sw.Elapsed.TotalSeconds;
            double gflops = 2.0 * n * n * n / seconds / 1e9;
            measurements.Add(gflops);
        }

        // Get output for verification
        var outputData = bufC.GetAsArray1D();

        var result = new BenchmarkResult
        {
            SuiteName = Name,
            BenchmarkName = $"Tiled {n}x{n}",
            DeviceName = profile.Name,
            MetricName = $"{n}x{n}",
            Unit = "GFLOPS",
        };
        result.ComputeStats(measurements);

        return (result, outputData);
    }

    private static bool VerifyArrays(float[] reference, float[] test, float epsilon, int totalElements)
    {
        if (reference.Length != test.Length) return false;

        // Check every 1000th element (or all elements if matrix is small)
        int step = totalElements <= 1024 ? 1 : 1000;

        for (int i = 0; i < reference.Length; i += step)
        {
            if (MathF.Abs(reference[i] - test[i]) > epsilon)
                return false;
        }
        return true;
    }

    private static int[] GetSizes(BenchmarkOptions options)
    {
        // If --size is specified, use only that size
        if (options.MatrixSize.HasValue)
            return [options.MatrixSize.Value];

        if (options.Quick)
            return [512, 1024, 2048];

        // Normal mode
        return options.Full
            ? [1024, 2048, 4096, 8192]
            : [1024, 2048, 4096];
    }

    private void RenderSummaryTable(List<BenchmarkResult> results, IReadOnlyList<DeviceProfile> profiles, int[] sizes)
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]Matrix Multiplication[/]");

        table.AddColumn(new TableColumn("[bold]Benchmark[/]"));
        foreach (var profile in profiles)
        {
            table.AddColumn(new TableColumn($"[{profile.Color} bold]{Markup.Escape(profile.Name)}[/]").RightAligned());
        }

        foreach (int n in sizes)
        {
            string[] kernelTypes = [$"Naive {n}x{n}", $"Tiled {n}x{n}"];

            foreach (var kernelName in kernelTypes)
            {
                var rowValues = new List<string>();
                rowValues.Add($"[bold]{Markup.Escape(kernelName)}[/]");

                var deviceValues = new List<(int idx, double value, string unit, BenchmarkResult? result)>();

                int validIdx = 0;
                foreach (var profile in profiles)
                {
                    var result = results.FirstOrDefault(r =>
                        r.BenchmarkName == kernelName && r.DeviceName == profile.Name);

                    if (result == null)
                    {
                        deviceValues.Add((-1, 0, "", null));
                    }
                    else if (result.IsError)
                    {
                        deviceValues.Add((-1, 0, result.Unit, result));
                    }
                    else
                    {
                        deviceValues.Add((validIdx, result.Best, result.Unit, result));
                        validIdx++;
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
                        bool isSkipped = dv.result.ErrorMessage?.StartsWith("Skipped") == true;
                        rowValues.Add(isSkipped ? "[dim]Skipped[/]" : "[red]FAILED[/]");
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

            // Add speedup row
            var speedupRow = new List<string>();
            speedupRow.Add("[dim italic]  Speedup (tiled/naive)[/]");

            foreach (var profile in profiles)
            {
                var naiveResult = results.FirstOrDefault(r =>
                    r.BenchmarkName == $"Naive {n}x{n}" && r.DeviceName == profile.Name && !r.IsError);
                var tiledResult = results.FirstOrDefault(r =>
                    r.BenchmarkName == $"Tiled {n}x{n}" && r.DeviceName == profile.Name && !r.IsError);

                if (naiveResult != null && tiledResult != null && naiveResult.Best > 0)
                {
                    double speedup = tiledResult.Best / naiveResult.Best;
                    speedupRow.Add($"[dim]{speedup:F2}x[/]");
                }
                else
                {
                    speedupRow.Add("[dim]N/A[/]");
                }
            }

            table.AddRow(speedupRow.ToArray());
        }

        AnsiConsole.Write(table);
    }

    private static void PrintInlineResult(BenchmarkResult result, DeviceProfile profile)
    {
        if (result.IsError)
        {
            AnsiConsole.MarkupLine($"      [red]{Markup.Escape(result.BenchmarkName)}: FAILED - {Markup.Escape(result.ErrorMessage ?? "Unknown error")}[/]");
        }
        else
        {
            AnsiConsole.MarkupLine(
                $"      [{profile.Color}]{Markup.Escape(result.BenchmarkName)}: {result.Best:F1} {result.Unit} (avg {result.Average:F1}, stddev {result.StdDev:F1})[/]");
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
}
