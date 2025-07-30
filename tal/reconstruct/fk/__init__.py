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

from tal.io.capture_data import NLOSCaptureData
import cupy as cp
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

        # Padding of the data
        N = data.sensor_grid_xyz.shape[0]
        M = data.H.shape[0]
        width = data.sensor_grid_xyz[-1,-1,0]
        range = data.delta_t*M

        data = data.H
        data = cp.asarray(data)

        z,y,x = cp.mgrid[-M:M, -N:N, -N:N]
        x = x/N
        y = y/N
        z = z/M

        grid_z = cp.tile(cp.linspace(0, 1, M, dtype=cp.float16)[:, cp.newaxis, cp.newaxis], (1, N, N))
        aux_data = cp.sqrt((data*grid_z)**2)

        t_data = cp.zeros((2*M, 2*N, 2*N))
        t_data[:M, :N, :N] = aux_data
        # FFT
        f_data = cp.fft.fftshift(cp.fft.fftn(t_data))

        # Stolt trick
        sqrt_term = cp.sqrt((N*range/(M*width*4))**2 * (x**2 + y**2) + z**2)
        f_vol = cp.interpn((z[:,0,0],y[0,:,0],x[0,0,:]),
                        f_data, 
                        cp.moveaxis(cp.array([sqrt_term, y, x]), 0,-1),
                        method = 'linear',
                        bounds_error = False,
                        fill_value=0)
        f_vol *= z>0
        f_vol *= cp.abs(z) / cp.max(sqrt_term)

        # IFFT
        t_vol = cp.fft.ifftn(cp.fft.ifftshift(f_vol))
        t_vol = cp.abs(t_vol)**2

        # Back to CPU
        t_vol = t_vol.get()  # Convert back to numpy array

        return t_vol[:M, :N, :N]
