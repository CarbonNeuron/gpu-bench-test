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
