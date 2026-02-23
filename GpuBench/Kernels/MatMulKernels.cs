using ILGPU;
using ILGPU.Runtime;

namespace GpuBench.Kernels;

public static class MatMulKernels
{
    public const int TileSize = 16;

    /// <summary>
    /// Naive matrix multiply: one thread per output element.
    /// Matrices are square (n x n) stored as row-major 1D arrays.
    /// Thread (row, col) computes the dot product of row from A and column from B.
    /// </summary>
    public static void NaiveMatMulKernel(
        Index2D index,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> c,
        int n)
    {
        int row = index.X;
        int col = index.Y;

        if (row >= n || col >= n)
            return;

        float sum = 0.0f;
        for (int k = 0; k < n; k++)
        {
            sum += a[row * n + k] * b[k * n + col];
        }

        c[row * n + col] = sum;
    }

    /// <summary>
    /// Tiled matrix multiply using shared memory with 16x16 tiles.
    /// Uses explicit grid/group indexing (no Index parameter).
    /// Launch with: kernel((gridDim, groupDim), a, b, c, n)
    /// where groupDim = (TileSize, TileSize) and gridDim = (ceil(n/TileSize), ceil(n/TileSize))
    /// </summary>
    public static void TiledMatMulKernel(
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> c,
        int n)
    {
        var tileA = SharedMemory.Allocate1D<float>(TileSize * TileSize);
        var tileB = SharedMemory.Allocate1D<float>(TileSize * TileSize);

        int row = Grid.GlobalIndex.X;
        int col = Grid.GlobalIndex.Y;

        int localRow = Group.IdxX;
        int localCol = Group.IdxY;

        float sum = 0.0f;

        // Number of tiles needed to cover the k dimension
        int numTiles = (n + TileSize - 1) / TileSize;

        for (int t = 0; t < numTiles; t++)
        {
            // Load tile from A: row from global, column from tile offset
            int aCol = t * TileSize + localCol;
            if (row < n && aCol < n)
                tileA[localRow * TileSize + localCol] = a[row * n + aCol];
            else
                tileA[localRow * TileSize + localCol] = 0.0f;

            // Load tile from B: row from tile offset, column from global
            int bRow = t * TileSize + localRow;
            if (bRow < n && col < n)
                tileB[localRow * TileSize + localCol] = b[bRow * n + col];
            else
                tileB[localRow * TileSize + localCol] = 0.0f;

            Group.Barrier();

            // Compute partial dot product for this tile
            for (int k = 0; k < TileSize; k++)
            {
                sum += tileA[localRow * TileSize + k] * tileB[k * TileSize + localCol];
            }

            Group.Barrier();
        }

        if (row < n && col < n)
        {
            c[row * n + col] = sum;
        }
    }
}
