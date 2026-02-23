using ILGPU;
using ILGPU.Runtime;

namespace GpuBench.Kernels;

public static class MemoryKernels
{
    // Sequential read: thread i reads element i
    public static void SequentialReadKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output)
    {
        output[index] = input[index];
    }

    // Random read: thread i reads from shuffled index
    public static void RandomReadKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> indices,
        ArrayView1D<float, Stride1D.Dense> output)
    {
        output[index] = input[indices[index]];
    }

    // Strided read: thread i reads element (i * stride) % maxIndex
    public static void StridedReadKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int stride,
        int maxIndex)
    {
        int readIdx = (index.X * stride) % maxIndex;
        output[index] = input[readIdx];
    }

    // Global memory repeated read: reads same element N times from global
    public static void GlobalRepeatedReadKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int repeats)
    {
        float sum = 0;
        for (int i = 0; i < repeats; i++)
            sum += input[index];
        output[index] = sum;
    }

    // Shared memory repeated read: loads into shared memory once, reads N times
    public static void SharedRepeatedReadKernel(
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int repeats)
    {
        var shared = SharedMemory.Allocate1D<float>(256);
        int localIdx = Group.IdxX;
        int globalIdx = Grid.GlobalIndex.X;

        if (localIdx < 256 && globalIdx < input.Length)
            shared[localIdx] = input[globalIdx];
        Group.Barrier();

        float sum = 0;
        if (localIdx < 256)
        {
            for (int i = 0; i < repeats; i++)
                sum += shared[localIdx];
        }

        if (globalIdx < output.Length)
            output[globalIdx] = sum;
    }
}
