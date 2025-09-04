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
        data = cp.asarray(np.transpose(data.H, (1, 2, 0)), dtype=float_dtype)  # Convert to cupy array with float16 precision
        
        # print(N, M, N * range / (M * width * 4))
        #

        # z, y, x = cp.mgrid[-M:M, -N:N, -N:N].astype(float_dtype)
        # z, y, x = z / M, y / N, x / N  # Normalize to [-1, 1] range

        # epsilon = 1e-8
        # sqrt_term = cp.sqrt((N*range/(M*width*4))**2 * (x**2 + y**2) + z**2)
        # grid_z = cp.tile(cp.linspace(0, 1, M, dtype=float_dtype)[:, cp.newaxis, cp.newaxis], (1, N, N))
        # data = cp.sqrt(data*grid_z**2).astype(complex_dtype)  

        # FFT
        t_data = cp.zeros((2*N, 2*N, 2*M), dtype=complex_dtype)  
        t_data[:N, :N, :M] = data
        t_data = cp.fft.fftshift(cp.fft.fftn(t_data))

        # Stolt trick
        threads_per_block = (8, 8, 8)
        blocks_per_grid = (
            (N + threads_per_block[0] - 1) // threads_per_block[0],
            (N + threads_per_block[1] - 1) // threads_per_block[1],
            (M + threads_per_block[2] - 1) // threads_per_block[2],
        )

        out_data = cp.zeros((N, N, M), dtype=complex_dtype)
        fk_numba[blocks_per_grid, threads_per_block](t_data, out_data, N * range / (M * width * 4))

        # # Stolt trick
        # f_vol = interpn((z[:,0,0], y[0,:,0], x[0,0,:]),
        #                 f_data, 
        #                 cp.moveaxis(cp.array([sqrt_term, y, x], dtype=float_dtype), 0,-1),
        #                 method = 'linear',
        #                 bounds_error = False,
        #                 fill_value=0)
        # f_vol *= z>0
        # f_vol *= cp.abs(z) / (sqrt_term + epsilon)

        # IFFT
        t_vol = cp.real(cp.fft.ifftn(cp.fft.ifftshift(out_data))).astype(float_dtype)
        t_vol = t_vol**2
        t_vol = t_vol.get()  # Convert back to numpy array

        return t_vol[:M, :N, :N]

@cuda.jit
def fk_numba(
    d_H_0,                 # Starting data
    d_H_1,                 # Output data        
    stolt_const
):
    # res_x, res_y, res_t = d_H_0.shape
    # epsilon = 1e-8

    v = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    l = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    s = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    
    # if t >= res_t or x >= res_x or y >= res_y:
    #     return
    
    # fx = 2.0 * float(x) / float(resolution[0]) - 1.0
    # fy = 2.0 * float(y) / float(resolution[1]) - 1.0
    # fz = 2.0 * float(t) / float(resolution[2]) - 1.0

    # sqrt_term = math.sqrt(stolt_const * (fx*fx + fy*fy + fz*fz))

    # ix = (fx + 1.0) * 0.5 * resolution[0]
    # iy = (fy + 1.0) * 0.5 * resolution[1]
    # iz = (fz + 1.0) * 0.5 * resolution[2]

    # x0 = int(math.floor(ix))
    # y0 = int(math.floor(iy))
    # z0 = int(math.floor(iz))
    # x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    # dx, dy, dz = ix - x0, iy - y0, iz - z0

    # x0 = max(0, min(x0, resolution[0] - 1))
    # y0 = max(0, min(y0, resolution[1] - 1))
    # z0 = max(0, min(z0, resolution[2] - 1))
    # x1 = max(0, min(x1, resolution[0] - 1))
    # y1 = max(0, min(y1, resolution[1] - 1))
    # z1 = max(0, min(z1, resolution[2] - 1))

    # c000 = d_H_0[x0, y0, z0]
    # c001 = d_H_0[x1, y0, z0]
    # c010 = d_H_0[x0, y1, z0]
    # c011 = d_H_0[x1, y1, z0]
    # c100 = d_H_0[x0, y0, z1]
    # c101 = d_H_0[x1, y0, z1]
    # c110 = d_H_0[x0, y1, z1]
    # c111 = d_H_0[x1, y1, z1]
    # c00 = c000 * (1 - dx) + c001 * dx
    # c01 = c010 * (1 - dx) + c011 * dx
    # c10 = c100 * (1 - dx) + c101 * dx
    # c11 = c110 * (1 - dx) + c111 * dx
    # c0 = c00 * (1 - dy) + c01 * dy
    # c1 = c10 * (1 - dy) + c11 * dy
    # c = c0 * (1 - dz) + c1 * dz
    # d = math.abs(fz) / (sqrt_term + epsilon)

    # d_H_1[x, y, t] = c * d
