using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace GpuBench.Models;

public sealed class DeviceProfile
{
    public required string Name { get; init; }
    public required string DeviceType { get; init; } // "CPU", "CUDA", "OpenCL"
    public required int DeviceIndex { get; init; }
    public int ComputeUnits { get; init; }
    public int MaxThreadsPerGroup { get; init; }
    public long MaxSharedMemoryPerGroup { get; init; }
    public long MemorySize { get; init; }
    public int WarpSize { get; init; }
    public int ClockRate { get; init; }
    public string? DriverVersion { get; init; }

    public string Color => DeviceType switch
    {
        "CUDA" => "green",
        "OpenCL" => "cyan",
        "CPU" => "yellow",
        _ => "white"
    };

    public bool IsCpu => DeviceType == "CPU";

    public static DeviceProfile FromAccelerator(Accelerator accelerator, int index)
    {
        var device = accelerator.Device;

        int clockRate = device switch
        {
            CudaDevice cuda => cuda.ClockRate,
            CLDevice cl => cl.ClockRate,
            _ => 0
        };

        string? driverVersion = device is CudaDevice cudaDev
            ? cudaDev.DriverVersion.ToString()
            : null;

        return new DeviceProfile
        {
            Name = device.Name,
            DeviceType = device.AcceleratorType switch
            {
                AcceleratorType.CPU => "CPU",
                AcceleratorType.Cuda => "CUDA",
                AcceleratorType.OpenCL => "OpenCL",
                _ => "Unknown"
            },
            DeviceIndex = index,
            ComputeUnits = device.NumMultiprocessors,
            MaxThreadsPerGroup = device.MaxNumThreadsPerGroup,
            MaxSharedMemoryPerGroup = device.MaxSharedMemoryPerGroup,
            MemorySize = device.MemorySize,
            WarpSize = device.WarpSize,
            ClockRate = clockRate,
            DriverVersion = driverVersion,
        };
    }
}
