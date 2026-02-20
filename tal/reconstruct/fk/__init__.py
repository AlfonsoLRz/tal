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

@author: Pablo Luesia-Lahoz, adapted from "Wave-Based Non-Line-of-Sight Imaging using Fast f-k Migration". Migrated to GPU by Alfonso López-Ruiz.
"""

from tal.io.capture_data import NLOSCaptureData
import cupy as cp
import numpy as np
import os

# Read CUDA kernel from file
parent_dir = os.path.abspath(os.path.dirname(__file__))
with open(f'{parent_dir}/fk.cu', 'r') as kernel_file:
    fk_kernel_source = kernel_file.read()

# Compile once
fk_kernel = cp.RawKernel(
    fk_kernel_source,
    "stoltKernel",
    options=(
        "--use_fast_math",
        # "-arch=compute_89",
    ),
)


# Python-side pipeline
def solve(data: NLOSCaptureData, downscale: int = 1) -> NLOSCaptureData.SingleReconstructionType:
    """
    Reconstruction using fk-migration (confocal only).
    """
    assert data.is_confocal(), \
        "Data must be confocal to use fk-migration with y-tal"

    # TODO: implement non-confocal variant
    if data.is_confocal():
        from tal.config import get_resources
        if downscale is not None and downscale > 1:
            data.spatial_downscale(downscale)

        # Types, tho this is not maintained yet because of CUDA kernel
        float_dtype, complex_dtype = cp.float32, cp.complex64

        # Dimensions
        N = data.sensor_grid_xyz.shape[0]   # spatial (x/y)
        M = data.H.shape[0]                 # temporal
        width = float_dtype(data.sensor_grid_xyz[-1, -1, 0])
        time_range = float_dtype(data.delta_t * M)

        # Move transient to GPU directly as complex64 (only real part is different than zero)
        h_gpu = cp.asarray(data.H, dtype=complex_dtype)

        t_data = cp.zeros((2 * M, 2 * N, 2 * N), dtype=complex_dtype)
        t_data[:M, :N, :N] = h_gpu

        # Forward 3D FFT (in-place)
        t_data = cp.fft.fftn(t_data)

        # Stolt parameters
        stolt_const = N * time_range / (M * width * 4.0)
        stolt_const_sq = float_dtype(stolt_const * stolt_const)

        # Output of Stolt interpolation
        out_data = cp.zeros_like(t_data)

        # Shapes and inverse resolutions
        Z, Y, X = t_data.shape  # Z = 2*M, Y = 2*N, X = 2*N
        invX, invY, invZ = np.float32(1.0 / float(X)), np.float32(1.0 / float(Y)), np.float32(1.0 / float(Z))
        shiftX, shiftY, shiftZ = np.int32(X // 2), np.int32(Y // 2), np.int32(Z // 2)

        threads = (8, 8, 8)
        blocks = (
            (X + threads[0] - 1) // threads[0],
            (Y + threads[1] - 1) // threads[1],
            (Z // 2 + threads[2] - 1) // threads[2],  # process upper half (as in C++)
        )

        fk_kernel(
            blocks,
            threads,
            (
                t_data,                
                out_data,              
                np.int32(X), np.int32(Y), np.int32(Z),
                invX, invY, invZ,               # For saving computations
                shiftX, shiftY, shiftZ,         # For avoiding computing stolt on the whole grid
                np.float32(stolt_const_sq),
            ),
        )
        # cp.cuda.runtime.deviceSynchronize()

        # Inverse FFT back to time/space domain
        out_data = cp.fft.ifftn(out_data)
        out_data = cp.real(out_data).astype(float_dtype)
        out_data = out_data ** 2
        out_data = out_data.get()
        out_data = np.transpose(out_data, (1, 2, 0))  # swap x/y to match original orientation

        return out_data[:N, :N, :M]