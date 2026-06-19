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
    assert data.is_confocal(), "Data must be confocal to use fk-migration with y-tal"

    # TODO: implement non-confocal variant
    if data.is_confocal():
        if downscale is not None and downscale > 1:
            data.spatial_downscale(downscale)

        # Types, tho this is not maintained yet because of CUDA kernel
        float_dtype, complex_dtype = cp.float32, cp.complex64

        # Dimensions
        nt, sx, sy = data.H.shape
        assert data.sensor_grid_xyz.shape[:2] == (sx, sy), \
            'H does not match with sensor_grid_xyz'
        grid_x = data.sensor_grid_xyz[..., 0]
        grid_y = data.sensor_grid_xyz[..., 1]
        half_x = np.float32((np.max(grid_x) - np.min(grid_x)) * 0.5)
        half_y = np.float32((np.max(grid_y) - np.min(grid_y)) * 0.5)
        assert half_x > 0.0 and half_y > 0.0, \
            'Could not infer relay wall size for fk-migration'
        time_range = np.float32(data.delta_t * nt)

        # Match FasterNLOS' FK padding: later time bins get a linear ramp.
        h_gpu = cp.asarray(data.H, dtype=float_dtype)
        temporal_scale = cp.arange(nt, dtype=float_dtype).reshape(nt, 1, 1)
        temporal_scale /= np.float32(nt)

        t_data = cp.zeros((2 * nt, 2 * sx, 2 * sy), dtype=complex_dtype)
        t_data[:nt, :sx, :sy] = h_gpu * temporal_scale

        # Forward 3D FFT (in-place)
        t_data = cp.fft.fftn(t_data)

        # Stolt parameters
        # t_data is laid out as (t, x, y), but the CUDA kernel's X axis
        # corresponds to the physical y dimension and Y to physical x.
        stolt_const_kernel_x = sy * time_range / (nt * half_y * 4.0)
        stolt_const_kernel_y = sx * time_range / (nt * half_x * 4.0)
        stolt_const_x_sq = np.float32(stolt_const_kernel_x * stolt_const_kernel_x)
        stolt_const_y_sq = np.float32(stolt_const_kernel_y * stolt_const_kernel_y)

        # Output of Stolt interpolation
        out_data = cp.zeros_like(t_data)

        # Shapes and inverse resolutions
        Z, Y, X = t_data.shape  # Z = 2*nt, Y = 2*sx, X = 2*sy
        invX, invY, invZ = np.float32(1.0 / float(X)), np.float32(1.0 / float(Y)), np.float32(1.0 / float(Z))
        shiftX, shiftY, shiftZ = np.int32(X // 2), np.int32(Y // 2), np.int32(Z // 2)

        threads = (8, 8, 8)
        blocks = (
            (Z // 2 + threads[0] - 1) // threads[0],  # z, upper half as in C++ implementation
            (Y + threads[1] - 1) // threads[1],
            (X + threads[2] - 1) // threads[2],
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
                stolt_const_x_sq,
                stolt_const_y_sq,
            ),
        )
        # cp.cuda.runtime.deviceSynchronize()

        # Inverse FFT back to time/space domain
        out_data = cp.fft.ifftn(out_data)
        out_data = cp.abs(out_data).astype(float_dtype)
        out_data = out_data.get()
        out_data = np.transpose(out_data, (1, 2, 0))  # swap x/y to match original orientation

        return out_data[:sx, :sy, :nt]
