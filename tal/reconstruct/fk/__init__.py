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
        float_dtype, complex_dtype = cp.float16, cp.complex64

        # Padding of the data
        N = data.sensor_grid_xyz.shape[0]
        M = data.H.shape[0]
        width = data.sensor_grid_xyz[-1,-1,0]
        range = data.delta_t*M

        # Convert to cupy array with float16 precision
        data = cp.asarray(data.H, dtype=float_dtype)  # Convert to cupy array with float16 precision

        # FFT
        print(M, N, N)
        t_data = cp.zeros((2*M, 2*N, 2*N), dtype=complex_dtype)  
        t_data[:M, :N, :N] = data.astype(complex_dtype)
        t_data = cp.fft.fftshift(cp.fft.fftn(t_data))

        # Stolt trick
        threads_per_block = (16, 8, 8)
        blocks_per_grid = (
            (M + threads_per_block[0] - 1) // threads_per_block[0],
            (2 * N + threads_per_block[1] - 1) // threads_per_block[1],
            (2 * N + threads_per_block[2] - 1) // threads_per_block[2],
        )

        stolt_const = N * range / (M * width * 4)
        stolt_const_sq = stolt_const * stolt_const
        
        out_data = cp.zeros((2 * M, 2 * N, 2 * N), dtype=complex_dtype)
        fk_numba[blocks_per_grid, threads_per_block](t_data, out_data, stolt_const_sq, 2 * N, 2 * N, 2 * M, 1e-8)

        # IFFT
        t_vol = cp.real(cp.fft.ifftn(cp.fft.ifftshift(out_data))).astype(float_dtype)
        t_vol = t_vol**2
        t_vol = t_vol.get()  

        return t_vol[:M, :N, :N]

@cuda.jit
def fk_numba(
    d_H_0,                 # Starting data
    d_H_1,                 # Output data        
    stolt_const,
    X, Y, Z,
    epsilon
):
    z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + Z // 2
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    x = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    if z >= Z or y >= Y or x >= X:
        return
    
    fx = 2.0 * float(x) / float(X) - 1.0
    fy = 2.0 * float(y) / float(Y) - 1.0
    fz = 2.0 * float(z) / float(Z) - 1.0

    sqrt_term = math.sqrt(stolt_const * (fx*fx + fy*fy) + fz*fz)

    ix = (fx + 1.0) * 0.5 * X
    iy = (fy + 1.0) * 0.5 * Y
    iz = (sqrt_term + 1.0) * 0.5 * Z

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

    c000 = d_H_0[z0, y0, x0]
    c001 = d_H_0[z0, y0, x1]
    c010 = d_H_0[z0, y1, x0]
    c011 = d_H_0[z0, y1, x1]
    c100 = d_H_0[z1, y0, x0]
    c101 = d_H_0[z1, y0, x1]
    c110 = d_H_0[z1, y1, x0]
    c111 = d_H_0[z1, y1, x1]

    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx
    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy
    c = c0 * (1 - dz) + c1 * dz

    d = math.fabs(fz) / (sqrt_term + epsilon)
    d_H_1[z, y, x] = c * d
