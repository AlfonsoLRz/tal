"""
tal.reconstruct.fk
===================

Reconstruction using the fk-migration algorithm.
See "Wave-Based Non-Line-of-Sight Imaging using Fast f-k Migration"

This implementation is an alternative to the other bp, fbp and pf/pf_dev 
approaches. 

WARNING: the fk-migration demands a lot of memory usage. If you think you might
get memory errors, try downscaling the y-tal data or trim the latest temporal
data.
"""

from turtle import width
from tal.io.capture_data import NLOSCaptureData
import cupy as cp
import math
from numba import cuda
import numpy as np
from cupyx.scipy.interpolate import interpn


def solve(data: NLOSCaptureData) -> NLOSCaptureData.SingleReconstructionType:
    """
    See module description of tal.reconstruct.fbp

    data
        See tal.io.read_capture
    """
    assert data.is_confocal(), \
        "Data must be confocal to use fk-migration with y-tal"
    # TODO: implement non confocal approach from "Wave-Based Non-Line-of-Sight Imaging using Fast f-k Migration"
    if data.is_confocal():
        from tal.config import get_resources
        downscale = get_resources().downscale
        if downscale is not None and downscale > 1:
            data.downscale(downscale)

        # Data types to reduce memory usage
        float_dtype, complex_dtype = cp.float32, cp.complex64

        # Padding of the data
        N = data.sensor_grid_xyz.shape[0]
        M = data.H.shape[0]
        width = (float_dtype)(data.sensor_grid_xyz[-1,-1,0])
        range = (float_dtype)(data.delta_t*M)

        # Convert to cupy array with float16 precision
        data = cp.asarray(data.H, dtype=float_dtype)  # Convert to cupy array with float16 precision

        # FFT
        t_data = cp.zeros((2*M, 2*N, 2*N), dtype=complex_dtype)  
        t_data[:M, :N, :N] = data.astype(complex_dtype)
        t_data = cp.fft.fftn(t_data)

        # Stolt trick
        threads_per_block = (16, 4, 4)  # x,y,z
        blocks_per_grid = (
            (2 * N + threads_per_block[0] - 1) // threads_per_block[0],
            (2 * N + threads_per_block[1] - 1) // threads_per_block[1],
            (M     + threads_per_block[2] - 1) // threads_per_block[2],
        )

        stolt_const = N * range / (M * width * 4)
        stolt_const_sq = stolt_const * stolt_const
        
        out_data = cp.zeros((2 * M, 2 * N, 2 * N), dtype=complex_dtype)
        
        invX = 1.0 / (2*N)
        invY = 1.0 / (2*N)
        invZ = 1.0 / (2*M)
        fk_kernel = cp.RawKernel(fk_kernel_source, "stoltKernel")
        fk_kernel(
            blocks_per_grid, threads_per_block,
            (t_data, out_data, np.float32(stolt_const),
            np.int32(2*N), np.int32(2*N), np.int32(2*M),
            np.float32(invX), np.float32(invY), np.float32(invZ),
            np.int32(N//2), np.int32(N//2), np.int32(M//2),
            stolt_const_sq)
        )

        # IFFT
        out_data = cp.real(cp.fft.ifftn(out_data)).astype(float_dtype)
        out_data = out_data**2
        out_data = out_data.get()

        return out_data[:M, :N, :N]

@cuda.jit(fastmath=True)
def fk_numba(
    d_input, d_output,                
    stolt_const,
    X, Y, Z,
    shift_X, shift_Y, shift_Z,
    epsilon
):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    z = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z + shift_Z

    if z >= Z or y >= Y or x >= X:
        return
    
    fx = 2.0 * float(x) / float(X) - 1.0
    fy = 2.0 * float(y) / float(Y) - 1.0
    fz = 2.0 * float(z) / float(Z) - 1.0

    sqrt_term = math.sqrt(stolt_const * (fx*fx + fy*fy) + fz*fz)

    ix = (fx + 1.0) * 0.5 * X
    iy = (fy + 1.0) * 0.5 * Y
    iz = (sqrt_term + 1.0) * 0.5 * Z

    ix = (ix + shift_X) % X
    iy = (iy + shift_Y) % Y
    iz = (iz + shift_Z) % Z

    x0 = int(math.floor(ix))
    y0 = int(math.floor(iy))
    z0 = int(math.floor(iz))
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    dx, dy, dz = ix - x0, iy - y0, iz - z0

    x0 = max(0, min(x0, X - 1))
    y0 = max(0, min(y0, Y - 1))
    z0 = max(0, min(z0, Z - 1))
    x1 = max(0, min(x1, X - 1))
    y1 = max(0, min(y1, Y - 1))
    z1 = max(0, min(z1, Z - 1))

    c000 = d_input[z0, y0, x0]
    c001 = d_input[z0, y0, x1]
    c010 = d_input[z0, y1, x0]
    c011 = d_input[z0, y1, x1]
    c100 = d_input[z1, y0, x0]
    c101 = d_input[z1, y0, x1]
    c110 = d_input[z1, y1, x0]
    c111 = d_input[z1, y1, x1]

    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx
    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy
    c = c0 * (1 - dz) + c1 * dz

    d = math.fabs(fz) / (sqrt_term + epsilon)

    x = (x + shift_X) % X
    y = (y + shift_Y) % Y
    z = (z + shift_Z) % Z

    d_output[z, y, x] = c * d

fk_kernel_source = r"""
#include <cupy/complex.cuh>

typedef complex<float> complex_t;

#define EPS      0.00001f
#define FLT_MAX  3.402823466e+38F

__device__ complex_t complexMulScalar(complex_t a, float s) {
    return complex_t(a.real() * s, a.imag() * s);
}

__device__ complex_t complexLerp(complex_t a, complex_t b, float t) {
    return complex_t(
        a.real() + (b.real() - a.real()) * t,
        a.imag() + (b.imag() - a.imag()) * t
    );
}

__device__ float safeRCP(float x) {
    if (x > EPS || x < -EPS)
        return 1.0f / x;
    return x >= 0 ? FLT_MAX : -FLT_MAX;
}

__device__ __forceinline__
int idx3d(int x, int y, int z, int X, int Y, int Z) {
    return x + X * (y + Y * z);
}

extern "C" __global__
void stoltKernel(
        const complex_t* __restrict__ H,
        complex_t* __restrict__ result,
        int X, int Y, int Z,
        float invX, float invY, float invZ,
        int shiftX, int shiftY, int shiftZ,
        float stoltConst
    )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= X || y >= Y || z >= Z) return;

    float fx = 2.0f * (x * invX) - 1.0f;
    float fy = 2.0f * (y * invY) - 1.0f;
    float fz = 2.0f * (z * invZ) - 1.0f;

    float sqrt_term = sqrtf(stoltConst * (fx * fx + fy * fy) + fz * fz);

    float ix = (fx + 1.0f) * 0.5f * X;
    float iy = (fy + 1.0f) * 0.5f * Y;
    float iz = (sqrt_term + 1.0f) * 0.5f * Z;

    int x0 = max(0, min((int)floorf(ix), X - 1));
    int y0 = max(0, min((int)floorf(iy), Y - 1));
    int z0 = max(0, min((int)floorf(iz), Z - 1));

    int x1 = min(x0 + 1, X - 1);
    int y1 = min(y0 + 1, Y - 1);
    int z1 = min(z0 + 1, Z - 1);

    float dx = ix - x0;
    float dy = iy - y0;
    float dz = iz - z0;

    complex_t c000 = H[idx3d(x0, y0, z0, X, Y, Z)];
    complex_t c100 = H[idx3d(x1, y0, z0, X, Y, Z)];
    complex_t c010 = H[idx3d(x0, y1, z0, X, Y, Z)];
    complex_t c110 = H[idx3d(x1, y1, z0, X, Y, Z)];
    complex_t c001 = H[idx3d(x0, y0, z1, X, Y, Z)];
    complex_t c101 = H[idx3d(x1, y0, z1, X, Y, Z)];
    complex_t c011 = H[idx3d(x0, y1, z1, X, Y, Z)];
    complex_t c111 = H[idx3d(x1, y1, z1, X, Y, Z)];

    complex_t c00 = complexLerp(c000, c100, dx);
    complex_t c10 = complexLerp(c010, c110, dx);
    complex_t c01 = complexLerp(c001, c101, dx);
    complex_t c11 = complexLerp(c011, c111, dx);

    complex_t c0 = complexLerp(c00, c10, dy);
    complex_t c1 = complexLerp(c01, c11, dy);
    complex_t res = complexLerp(c0, c1, dz);

    result[idx3d(x, y, z, X, Y, Z)] = complexMulScalar(res, fabsf(fz) * safeRCP(sqrt_term));
}
"""

fk_kernel = cp.RawKernel(fk_kernel_source, "fk_kernel")

