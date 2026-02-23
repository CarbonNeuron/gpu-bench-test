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
