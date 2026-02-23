using GpuBench.Models;
using Spectre.Console;

namespace GpuBench.Rendering;

public static class ResultsRenderer
{
    /// <summary>
    /// Renders ONE summary table at the end of a full run with key metrics per suite.
    /// </summary>
    public static void RenderSummaryTable(List<BenchmarkResult> allResults, IReadOnlyList<DeviceProfile> profiles)
    {
        if (allResults.Count == 0 || profiles.Count == 0) return;

        AnsiConsole.Write(new Rule("[bold blue]Overall Summary[/]").RuleStyle("blue"));
        AnsiConsole.WriteLine();

        var table = new Table()
            .Border(TableBorder.Rounded)
            .Title("[bold]Key Metrics Summary[/]");

        table.AddColumn(new TableColumn("[bold]Metric[/]"));
        foreach (var profile in profiles)
        {
            table.AddColumn(new TableColumn($"[{profile.Color} bold]{Markup.Escape(profile.Name)}[/]").RightAligned());
        }

        // Define rows: (display label, suite filter, benchmark filter, unit, higherIsBetter, format)
        var metricRows = new List<(string label, Func<BenchmarkResult, bool> filter, bool higherIsBetter, string format)>
        {
            ("FP32 (GFLOPS)",
                r => r.SuiteName == "compute" && r.BenchmarkName == "FP32 Throughput",
                true, "N1"),
            ("FP64 (GFLOPS)",
                r => r.SuiteName == "compute" && r.BenchmarkName == "FP64 Throughput",
                true, "N1"),
            ("Int32 (GOPS)",
                r => r.SuiteName == "compute" && r.BenchmarkName == "Int32 Throughput",
                true, "N1"),
            ("MatMul Tiled (GFLOPS)",
                r => r.SuiteName == "matmul" && r.BenchmarkName.StartsWith("Tiled"),
                true, "N1"),
            ("Memory H\u2192D (GB/s)",
                r => r.SuiteName == "memory" && r.BenchmarkName == "H\u2192D Transfer",
                true, "N1"),
            ("Memory D\u2192H (GB/s)",
                r => r.SuiteName == "memory" && r.BenchmarkName == "D\u2192H Transfer",
                true, "N1"),
            ("Kernel Launch (\u00b5s)",
                r => r.SuiteName == "latency" && r.BenchmarkName == "Kernel Launch",
                false, "N1"),
        };

        foreach (var (label, filter, higherIsBetter, format) in metricRows)
        {
            var rowValues = new List<string> { $"[bold]{Markup.Escape(label)}[/]" };

            var deviceValues = new List<(double value, bool hasValue, DeviceProfile profile)>();

            foreach (var profile in profiles)
            {
                // Find matching results for this device
                var matching = allResults
                    .Where(r => filter(r) && r.DeviceName == profile.Name && !r.IsError)
                    .ToList();

                if (matching.Count == 0)
                {
                    deviceValues.Add((0, false, profile));
                }
                else
                {
                    // For metrics with multiple sizes (matmul, memory), use the largest size result
                    var best = matching.OrderByDescending(r => r.Best).First();
                    deviceValues.Add((best.Best, true, profile));
                }
            }

            // Find the winner value
            double winnerValue = 0;
            bool hasAnyValue = deviceValues.Any(d => d.hasValue);

            if (hasAnyValue)
            {
                var validValues = deviceValues.Where(d => d.hasValue).Select(d => d.value);
                winnerValue = higherIsBetter ? validValues.Max() : validValues.Min();
            }

            foreach (var (value, hasValue, profile) in deviceValues)
            {
                if (!hasValue)
                {
                    rowValues.Add("[dim]\u2014[/]");
                }
                else
                {
                    string valueStr = value.ToString(format);
                    bool isWinner = Math.Abs(value - winnerValue) < 0.01 &&
                                    deviceValues.Count(d => d.hasValue) > 1;

                    rowValues.Add(isWinner ? $"[bold]{valueStr}[/]" : valueStr);
                }
            }

            table.AddRow(rowValues.ToArray());
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();

        // Render bar charts for key metrics
        RenderKeyBarCharts(allResults, profiles);
    }

    /// <summary>
    /// Renders bar charts for the most interesting metrics.
    /// </summary>
    private static void RenderKeyBarCharts(List<BenchmarkResult> allResults, IReadOnlyList<DeviceProfile> profiles)
    {
        // FP32 bar chart
        var fp32Items = new List<(string label, double value, string color)>();
        foreach (var profile in profiles)
        {
            var result = allResults.FirstOrDefault(r =>
                r.SuiteName == "compute" && r.BenchmarkName == "FP32 Throughput" &&
                r.DeviceName == profile.Name && !r.IsError);
            if (result != null)
                fp32Items.Add((profile.Name, result.Best, profile.Color));
        }

        if (fp32Items.Count > 0)
            RenderBarChart("FP32 Throughput", "GFLOPS", fp32Items);

        // MatMul Tiled bar chart (largest size)
        var matmulItems = new List<(string label, double value, string color)>();
        foreach (var profile in profiles)
        {
            var results = allResults
                .Where(r => r.SuiteName == "matmul" && r.BenchmarkName.StartsWith("Tiled") &&
                            r.DeviceName == profile.Name && !r.IsError)
                .OrderByDescending(r => r.Best)
                .FirstOrDefault();
            if (results != null)
                matmulItems.Add((profile.Name, results.Best, profile.Color));
        }

        if (matmulItems.Count > 0)
            RenderBarChart("MatMul Tiled (best)", "GFLOPS", matmulItems);
    }

    /// <summary>
    /// Renders a horizontal bar chart for visual comparison.
    /// </summary>
    public static void RenderBarChart(string title, string unit, IReadOnlyList<(string label, double value, string color)> items)
    {
        if (items.Count == 0) return;

        var chart = new BarChart()
            .Width(72)
            .Label($"[bold]{Markup.Escape(title)} ({Markup.Escape(unit)})[/]");

        foreach (var (label, value, color) in items)
        {
            chart.AddItem(Markup.Escape(label), Math.Max(value, 0.01), ParseColor(color));
        }

        AnsiConsole.Write(chart);
        AnsiConsole.WriteLine();
    }

    /// <summary>
    /// Renders a winner panel for a specific category.
    /// </summary>
    public static void RenderWinnerPanel(
        string category,
        string winnerName,
        string winnerType,
        double value,
        string unit,
        IReadOnlyList<(string name, double ratio)> comparisons)
    {
        var comparisonParts = new List<string>();
        foreach (var (name, ratio) in comparisons)
        {
            if (ratio >= 1.5)
                comparisonParts.Add($"{ratio:F0}x faster than {name}");
            else if (ratio > 1.0)
                comparisonParts.Add($"{ratio:F1}x faster than {name}");
        }

        string comparisonText = comparisonParts.Count > 0
            ? $"\n  {string.Join(", ", comparisonParts)}"
            : "";

        string content = $"  [bold]{Markup.Escape(winnerName)}[/] ({Markup.Escape(winnerType)}) \u2014 {value:N1} {Markup.Escape(unit)}{comparisonText}";

        var panel = new Panel(content)
            .Header($"[bold green] {Markup.Escape(category)} [/]")
            .Border(BoxBorder.Rounded)
            .BorderStyle(Style.Parse("green"))
            .Padding(1, 0);

        AnsiConsole.Write(panel);
        AnsiConsole.WriteLine();
    }

    /// <summary>
    /// Determines the overall champion across all categories and renders winner panels.
    /// </summary>
    public static void RenderOverallChampion(List<BenchmarkResult> allResults, IReadOnlyList<DeviceProfile> profiles)
    {
        if (allResults.Count == 0 || profiles.Count == 0) return;

        // Define categories to evaluate
        var categories = new List<(string name, Func<BenchmarkResult, bool> filter, bool higherIsBetter, string unit)>
        {
            ("Compute Champion (FP32)",
                r => r.SuiteName == "compute" && r.BenchmarkName == "FP32 Throughput",
                true, "GFLOPS"),
            ("MatMul Champion",
                r => r.SuiteName == "matmul" && r.BenchmarkName.StartsWith("Tiled"),
                true, "GFLOPS"),
            ("Memory Bandwidth Champion",
                r => r.SuiteName == "memory" && r.BenchmarkName == "H\u2192D Transfer",
                true, "GB/s"),
            ("Lowest Latency",
                r => r.SuiteName == "latency" && r.BenchmarkName == "Kernel Launch",
                false, "\u00b5s"),
        };

        var winCounts = new Dictionary<string, int>();
        foreach (var profile in profiles)
            winCounts[profile.Name] = 0;

        foreach (var (categoryName, filter, higherIsBetter, unit) in categories)
        {
            var candidates = new List<(DeviceProfile profile, double value)>();

            foreach (var profile in profiles)
            {
                var matching = allResults
                    .Where(r => filter(r) && r.DeviceName == profile.Name && !r.IsError)
                    .ToList();

                if (matching.Count > 0)
                {
                    double bestValue = higherIsBetter
                        ? matching.Max(r => r.Best)
                        : matching.Min(r => r.Best);
                    candidates.Add((profile, bestValue));
                }
            }

            if (candidates.Count == 0) continue;

            // Find winner
            var winner = higherIsBetter
                ? candidates.OrderByDescending(c => c.value).First()
                : candidates.OrderBy(c => c.value).First();

            winCounts[winner.profile.Name]++;

            // Build comparisons
            var comparisons = new List<(string name, double ratio)>();
            foreach (var (profile, value) in candidates)
            {
                if (profile.Name == winner.profile.Name) continue;

                double ratio = higherIsBetter
                    ? winner.value / Math.Max(value, 0.001)
                    : value / Math.Max(winner.value, 0.001);

                comparisons.Add((profile.Name, ratio));
            }

            RenderWinnerPanel(categoryName, winner.profile.Name, winner.profile.DeviceType,
                winner.value, unit, comparisons);
        }

        // Overall champion
        if (winCounts.Count > 0)
        {
            var overallWinner = winCounts.OrderByDescending(kvp => kvp.Value).First();
            var overallProfile = profiles.FirstOrDefault(p => p.Name == overallWinner.Key);

            if (overallProfile != null)
            {
                string wins = $"{overallWinner.Value} of {categories.Count(c => allResults.Any(r => c.filter(r) && !r.IsError))} categories";

                var panel = new Panel($"  [bold]{Markup.Escape(overallProfile.Name)}[/] ({Markup.Escape(overallProfile.DeviceType)})\n  Won {wins}")
                    .Header("[bold yellow] Overall Champion [/]")
                    .Border(BoxBorder.Double)
                    .BorderStyle(Style.Parse("yellow"))
                    .Padding(1, 0);

                AnsiConsole.Write(panel);
                AnsiConsole.WriteLine();
            }
        }
    }

    private static Color ParseColor(string color)
    {
        return color.ToLowerInvariant() switch
        {
            "green" => Color.Green,
            "cyan" => Color.Cyan1,
            "yellow" => Color.Yellow,
            "red" => Color.Red,
            "blue" => Color.Blue,
            "white" => Color.White,
            _ => Color.Grey,
        };
    }
}
