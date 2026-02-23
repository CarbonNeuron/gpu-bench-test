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
