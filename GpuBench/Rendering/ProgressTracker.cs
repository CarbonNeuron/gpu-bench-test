using Spectre.Console;

namespace GpuBench.Rendering;

public static class ProgressTracker
{
    public static void RunWithStatus(string message, Action action)
    {
        AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .SpinnerStyle(Style.Parse("blue"))
            .Start(message, ctx => action());
    }

    public static T RunWithStatus<T>(string message, Func<T> func)
    {
        T result = default!;
        AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .SpinnerStyle(Style.Parse("blue"))
            .Start(message, ctx => { result = func(); });
        return result;
    }
}
