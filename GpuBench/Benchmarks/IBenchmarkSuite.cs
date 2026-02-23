using GpuBench.Models;
using ILGPU.Runtime;

namespace GpuBench.Benchmarks;

public interface IBenchmarkSuite
{
    string Name { get; }
    string Description { get; }
    bool SupportsDevice(DeviceProfile device);
    List<BenchmarkResult> Run(IReadOnlyList<Accelerator> accelerators, IReadOnlyList<DeviceProfile> profiles, BenchmarkOptions options);
}
