using System.Text;
using Spectre.Console.Cli;

namespace GpuBench;

public static class Program
{
    public static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        var app = new CommandApp<BenchmarkCommand>();
        app.Configure(config =>
        {
            config.SetApplicationName("gpubench");
            config.SetApplicationVersion("1.0.0");
        });
        return app.Run(args);
    }
}
