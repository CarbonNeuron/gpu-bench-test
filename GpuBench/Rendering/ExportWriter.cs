using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using GpuBench.Models;

namespace GpuBench.Rendering;

public static class ExportWriter
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
    };

    /// <summary>
    /// Exports benchmark results to a JSON file.
    /// </summary>
    public static void ExportJson(string path, List<BenchmarkResult> results, IReadOnlyList<DeviceProfile> profiles)
    {
        var export = new ExportDocument
        {
            Timestamp = DateTime.UtcNow.ToString("O"),
            System = new SystemInfo
            {
                Os = RuntimeInformation.OSDescription,
                MachineName = Environment.MachineName,
                RuntimeVersion = RuntimeInformation.FrameworkDescription,
            },
            Devices = profiles.Select(p => new DeviceExport
            {
                Name = p.Name,
                Type = p.DeviceType,
                DeviceIndex = p.DeviceIndex,
                ComputeUnits = p.ComputeUnits,
                MaxThreadsPerGroup = p.MaxThreadsPerGroup,
                MaxSharedMemoryPerGroup = p.MaxSharedMemoryPerGroup,
                MemorySize = p.MemorySize,
                WarpSize = p.WarpSize,
                ClockRate = p.ClockRate,
                DriverVersion = p.DriverVersion,
            }).ToList(),
            Results = results.Select(r => new ResultExport
            {
                Suite = r.SuiteName,
                Benchmark = r.BenchmarkName,
                Device = r.DeviceName,
                Metric = r.MetricName,
                Unit = r.Unit,
                Best = r.Best,
                Average = r.Average,
                Worst = r.Worst,
                StdDev = r.StdDev,
                Verification = r.Verification.ToString(),
                Error = r.ErrorMessage,
            }).ToList(),
        };

        var json = JsonSerializer.Serialize(export, JsonOptions);
        File.WriteAllText(path, json);
    }

    /// <summary>
    /// Exports benchmark results to a Markdown file.
    /// </summary>
    public static void ExportMarkdown(string path, List<BenchmarkResult> results, IReadOnlyList<DeviceProfile> profiles)
    {
        var sb = new StringBuilder();

        sb.AppendLine("# GpuBench Results");
        sb.AppendLine();
        sb.AppendLine($"**Date:** {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
        sb.AppendLine($"**Machine:** {Environment.MachineName}");
        sb.AppendLine($"**OS:** {RuntimeInformation.OSDescription}");
        sb.AppendLine($"**Runtime:** {RuntimeInformation.FrameworkDescription}");
        sb.AppendLine();

        // Device info table
        sb.AppendLine("## Devices");
        sb.AppendLine();
        sb.AppendLine("| # | Name | Type | Compute Units | Max Threads/Group | Memory | Warp Size | Clock (MHz) |");
        sb.AppendLine("|---|------|------|---------------|-------------------|--------|-----------|-------------|");

        foreach (var p in profiles)
        {
            sb.AppendLine($"| {p.DeviceIndex} | {EscapeMarkdown(p.Name)} | {p.DeviceType} | {p.ComputeUnits} | {p.MaxThreadsPerGroup:N0} | {FormatBytes(p.MemorySize)} | {p.WarpSize} | {p.ClockRate} |");
        }

        sb.AppendLine();

        // Per-suite results tables
        var suites = results.Select(r => r.SuiteName).Distinct().ToList();

        foreach (var suite in suites)
        {
            var suiteResults = results.Where(r => r.SuiteName == suite).ToList();
            var benchmarkNames = suiteResults.Select(r => r.BenchmarkName).Distinct().ToList();
            var deviceNames = suiteResults.Select(r => r.DeviceName).Distinct().ToList();

            sb.AppendLine($"## {CapitalizeSuite(suite)}");
            sb.AppendLine();

            // Build header
            sb.Append("| Benchmark |");
            foreach (var device in deviceNames)
                sb.Append($" {EscapeMarkdown(device)} |");
            sb.AppendLine();

            sb.Append("|-----------|");
            foreach (var _ in deviceNames)
                sb.Append("-----------|");
            sb.AppendLine();

            foreach (var benchName in benchmarkNames)
            {
                sb.Append($"| {EscapeMarkdown(benchName)} |");

                foreach (var device in deviceNames)
                {
                    var result = suiteResults.FirstOrDefault(r =>
                        r.BenchmarkName == benchName && r.DeviceName == device);

                    if (result == null)
                    {
                        sb.Append(" N/A |");
                    }
                    else if (result.IsError)
                    {
                        sb.Append($" {EscapeMarkdown(result.ErrorMessage ?? "FAILED")} |");
                    }
                    else
                    {
                        sb.Append($" {result.Best:F2} {result.Unit} |");
                    }
                }

                sb.AppendLine();
            }

            sb.AppendLine();
        }

        // Summary section
        sb.AppendLine("## Summary");
        sb.AppendLine();

        var summaryMetrics = new (string label, Func<BenchmarkResult, bool> filter, bool higherIsBetter)[]
        {
            ("FP32 Throughput", r => r.SuiteName == "compute" && r.BenchmarkName == "FP32 Throughput", true),
            ("FP64 Throughput", r => r.SuiteName == "compute" && r.BenchmarkName == "FP64 Throughput", true),
            ("Int32 Throughput", r => r.SuiteName == "compute" && r.BenchmarkName == "Int32 Throughput", true),
            ("MatMul Tiled", r => r.SuiteName == "matmul" && r.BenchmarkName.StartsWith("Tiled"), true),
            ("Memory H\u2192D", r => r.SuiteName == "memory" && r.BenchmarkName == "H\u2192D Transfer", true),
            ("Kernel Launch", r => r.SuiteName == "latency" && r.BenchmarkName == "Kernel Launch", false),
        };

        foreach (var (label, filter, higherIsBetter) in summaryMetrics)
        {
            var candidates = new List<(string device, double value, string unit)>();

            foreach (var profile in profiles)
            {
                var matching = results
                    .Where(r => filter(r) && r.DeviceName == profile.Name && !r.IsError)
                    .ToList();

                if (matching.Count > 0)
                {
                    var best = higherIsBetter
                        ? matching.OrderByDescending(r => r.Best).First()
                        : matching.OrderBy(r => r.Best).First();
                    candidates.Add((profile.Name, best.Best, best.Unit));
                }
            }

            if (candidates.Count == 0) continue;

            var winner = higherIsBetter
                ? candidates.OrderByDescending(c => c.value).First()
                : candidates.OrderBy(c => c.value).First();

            sb.AppendLine($"- **{label}:** {EscapeMarkdown(winner.device)} \u2014 {winner.value:F2} {winner.unit}");
        }

        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine("*Generated by GpuBench*");

        File.WriteAllText(path, sb.ToString());
    }

    private static string EscapeMarkdown(string text)
    {
        return text.Replace("|", "\\|").Replace("*", "\\*").Replace("_", "\\_");
    }

    private static string CapitalizeSuite(string suite)
    {
        return suite switch
        {
            "compute" => "Compute Throughput",
            "memory" => "Memory Bandwidth",
            "matmul" => "Matrix Multiplication",
            "latency" => "Latency Tests",
            "patterns" => "Memory Access Patterns",
            _ => suite,
        };
    }

    private static string FormatBytes(long bytes)
    {
        // CPU accelerator reports long.MaxValue; show as "N/A" or system RAM label
        if (bytes >= long.MaxValue / 2) return "System RAM";
        if (bytes >= 1L << 30) return $"{bytes / (double)(1L << 30):F1} GB";
        if (bytes >= 1L << 20) return $"{bytes / (double)(1L << 20):F1} MB";
        if (bytes >= 1L << 10) return $"{bytes / (double)(1L << 10):F1} KB";
        return $"{bytes} B";
    }

    // ---- JSON export models ----

    private sealed class ExportDocument
    {
        public string Timestamp { get; set; } = "";
        public SystemInfo System { get; set; } = new();
        public List<DeviceExport> Devices { get; set; } = [];
        public List<ResultExport> Results { get; set; } = [];
    }

    private sealed class SystemInfo
    {
        public string Os { get; set; } = "";
        public string MachineName { get; set; } = "";
        public string RuntimeVersion { get; set; } = "";
    }

    private sealed class DeviceExport
    {
        public string Name { get; set; } = "";
        public string Type { get; set; } = "";
        public int DeviceIndex { get; set; }
        public int ComputeUnits { get; set; }
        public int MaxThreadsPerGroup { get; set; }
        public long MaxSharedMemoryPerGroup { get; set; }
        public long MemorySize { get; set; }
        public int WarpSize { get; set; }
        public int ClockRate { get; set; }
        public string? DriverVersion { get; set; }
    }

    private sealed class ResultExport
    {
        public string Suite { get; set; } = "";
        public string Benchmark { get; set; } = "";
        public string Device { get; set; } = "";
        public string Metric { get; set; } = "";
        public string Unit { get; set; } = "";
        public double Best { get; set; }
        public double Average { get; set; }
        public double Worst { get; set; }
        public double StdDev { get; set; }
        public string Verification { get; set; } = "NotApplicable";
        public string? Error { get; set; }
    }
}
