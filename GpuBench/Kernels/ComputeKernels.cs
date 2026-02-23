using ILGPU;
using ILGPU.Runtime;

namespace GpuBench.Kernels;

public static class ComputeKernels
{
    public const int FmaOpsPerThread = 1000;
    public const int FlopsPerFma = 2; // 1 mul + 1 add

    public static void Fp32FmaKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> output)
    {
        float val = index.X * 0.001f;
        for (int i = 0; i < FmaOpsPerThread; i++)
            val = val * 1.0001f + 0.0001f;
        output[index] = val;
    }

    public static void Fp64FmaKernel(Index1D index, ArrayView1D<double, Stride1D.Dense> output)
    {
        double val = index.X * 0.001;
        for (int i = 0; i < FmaOpsPerThread; i++)
            val = val * 1.0001 + 0.0001;
        output[index] = val;
    }

    public static void Int32FmaKernel(Index1D index, ArrayView1D<int, Stride1D.Dense> output)
    {
        int val = index.X;
        for (int i = 0; i < FmaOpsPerThread; i++)
            val = val * 3 + 7;
        output[index] = val;
    }

    public static void VectorAddKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c)
    {
        c[index] = a[index] + b[index];
    }

    public static void VectorFmaKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b, ArrayView1D<float, Stride1D.Dense> c)
    {
        c[index] = a[index] * b[index] + c[index];
    }
}
