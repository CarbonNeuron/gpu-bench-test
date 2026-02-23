using GpuBench.Benchmarks;
using GpuBench.Models;
using ILGPU;
using ILGPU.Runtime;
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
        var options = BenchmarkOptions.FromSettings(settings);

        AnsiConsole.Write(new Rule("[bold blue]GpuBench v1.0[/]").RuleStyle("blue"));
        AnsiConsole.WriteLine();

        using var ilContext = Context.Create(builder =>
        {
            builder.Default();
            builder.EnableAlgorithms();
        });

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
        new ComputeThroughput(),
        new MemoryBandwidth(),
        new MatrixMultiply(),
        new LatencyTests(),
        new MemoryPatterns(),
    ];
}
