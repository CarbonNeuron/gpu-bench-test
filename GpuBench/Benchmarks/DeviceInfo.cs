using GpuBench.Models;
using Spectre.Console;

namespace GpuBench.Benchmarks;

public static class DeviceInfo
{
    public static void Display(IReadOnlyList<DeviceProfile> profiles)
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]Detected Compute Devices[/]");

        table.AddColumn(new TableColumn("[bold]#[/]").RightAligned());
        table.AddColumn(new TableColumn("[bold]Name[/]"));
        table.AddColumn(new TableColumn("[bold]Type[/]"));
        table.AddColumn(new TableColumn("[bold]Compute Units[/]").RightAligned());
        table.AddColumn(new TableColumn("[bold]Max Threads/Group[/]").RightAligned());
        table.AddColumn(new TableColumn("[bold]Shared Mem/Group[/]").RightAligned());
        table.AddColumn(new TableColumn("[bold]Memory[/]").RightAligned());
        table.AddColumn(new TableColumn("[bold]Warp Size[/]").RightAligned());
        table.AddColumn(new TableColumn("[bold]Clock (MHz)[/]").RightAligned());

        foreach (var p in profiles)
        {
            table.AddRow(
                $"[{p.Color}]{p.DeviceIndex}[/]",
                $"[{p.Color} bold]{Markup.Escape(p.Name)}[/]",
                $"[{p.Color}]{p.DeviceType}[/]",
                $"[{p.Color}]{p.ComputeUnits}[/]",
                $"[{p.Color}]{p.MaxThreadsPerGroup:N0}[/]",
                $"[{p.Color}]{FormatBytes(p.MaxSharedMemoryPerGroup)}[/]",
                $"[{p.Color}]{FormatBytes(p.MemorySize)}[/]",
                $"[{p.Color}]{p.WarpSize}[/]",
                $"[{p.Color}]{p.ClockRate}[/]"
            );
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    private static string FormatBytes(long bytes)
    {
        if (bytes >= 1L << 30) return $"{bytes / (double)(1L << 30):F1} GB";
        if (bytes >= 1L << 20) return $"{bytes / (double)(1L << 20):F1} MB";
        if (bytes >= 1L << 10) return $"{bytes / (double)(1L << 10):F1} KB";
        return $"{bytes} B";
    }
}
