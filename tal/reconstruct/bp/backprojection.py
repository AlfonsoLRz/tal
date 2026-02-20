"""
Docstring for tal.reconstruct.bp.backprojection
"""

from tal.config import get_resources
from tal.log import log, LogLevel, TQDMLogRedirect
import cupy as cp
from numba import cuda
import numpy as np
import os

# Read CUDA kernel from file
parent_dir = os.path.abspath(os.path.dirname(__file__))
with open(f'{parent_dir}/backprojection.cu', 'r') as kernel_file:
    bp_kernel_source = kernel_file.read()

# Compile only once
bp_kernel = cp.RawKernel(
    bp_kernel_source,
    "backproject_kernel",
    options=(
        "--use_fast_math",
        # "-arch=compute_89",
    ),
)

def backproject(H_0, laser_grid_xyz, sensor_grid_xyz, volume_xyz, volume_xyz_shape,
                camera_system, t_accounts_first_and_last_bounces,
                t_start, delta_t, is_laser_paired_to_sensor,
                projector_focus=None,
                laser_xyz=None, sensor_xyz=None,
                compensate_invsq=False, progress=False):
    if camera_system.is_transient():
        log(LogLevel.WARNING, 'tal.reconstruct.bp: You have specified a time-resolved camera_system. '
            'The tal.reconstruct.bp implementation is better suited for time-gated systems. '
            'This will work, but you may want to check out tal.reconstruct.pf_dev for time-resolved reconstructions.')

    nt, nl, ns = H_0.shape
    nv, _ = volume_xyz.shape

    print(H_0.shape, volume_xyz.shape)
    print(nt, nl, ns, nv)

    if is_laser_paired_to_sensor:
        assert laser_grid_xyz.shape[0] == ns, 'H does not match with laser_grid_xyz'
    else:
        assert laser_grid_xyz.shape[0] == nl, 'H does not match with laser_grid_xyz'

    assert sensor_grid_xyz.shape[0] == ns, 'H does not match with sensor_grid_xyz'
    assert (not t_accounts_first_and_last_bounces or (laser_xyz is not None and sensor_xyz is not None)), \
        't_accounts_first_and_last_bounces requires laser_xyz and sensor_xyz'

    # Same as in the original code
    if camera_system.implements_projector():
        assert projector_focus is not None, 'projector_focus is required for this camera system'
        assert np.size(projector_focus) == 3, (
            'When using tal.reconstruct.bp, projector_focus must be a single 3D point. '
            'If you want to focus the illumination aperture at multiple points, '
            'please use tal.reconstruct.pf_dev instead or call tal.reconstruct.bp once per projector_focus.'
        )
        pf = np.asarray(projector_focus, dtype=np.float32).reshape(1, 3)
        projector_focus_arr = np.repeat(pf, nv, axis=0)  # [nv, 3]
    else:
        assert projector_focus is None, 'projector_focus must not be set for this camera system'
        projector_focus_arr = np.asarray(volume_xyz, dtype=np.float32)  # [nv, 3]

    # Transfer everything to GPU
    H_0_gpu = cp.asarray(H_0, dtype=cp.float32)
    laser_grid_gpu = cp.asarray(laser_grid_xyz, dtype=cp.float32)               # [nl,3] or [ns,3]
    sensor_grid_gpu = cp.asarray(sensor_grid_xyz, dtype=cp.float32)             # [ns,3]
    volume_gpu = cp.asarray(volume_xyz, dtype=cp.float32)                       # [nv,3]
    projector_focus_gpu = cp.asarray(projector_focus_arr, dtype=cp.float32)     # [nv,3]

    # If null, set to zero; TODO: check if it is possible to pass a null pointer to the CUDA kernel instead and avoid this step
    if laser_xyz is None:
        laser_xyz_arr = np.zeros(3, dtype=np.float32)
    else:
        laser_xyz_arr = np.asarray(laser_xyz, dtype=np.float32).reshape(3,)
    if sensor_xyz is None:
        sensor_xyz_arr = np.zeros(3, dtype=np.float32)
    else:
        sensor_xyz_arr = np.asarray(sensor_xyz, dtype=np.float32).reshape(3,)

    laser_xyz_gpu = cp.asarray(laser_xyz_arr, dtype=cp.float32)
    sensor_xyz_gpu = cp.asarray(sensor_xyz_arr, dtype=cp.float32)

    # output
    if camera_system.is_transient():
        H_1_gpu = cp.zeros((nt, nv), dtype=cp.float32)
    else:
        H_1_gpu = cp.zeros((nv,), dtype=cp.float32)

    threads = (8, 8, 8)
    blocks = (
        (nv + threads[0] - 1) // threads[0],
        (nl + threads[1] - 1) // threads[1],
        (ns + threads[2] - 1) // threads[2],
    )

    # Somehow, simply passing float32 causes issues; explicitly convert to cupy data types
    cp_delta = cp.float32(delta_t)
    cp_t_start = cp.float32(t_start)

    bp_kernel(
        blocks,
        threads,
        (
            H_0_gpu,
            H_1_gpu,
            laser_grid_gpu,
            sensor_grid_gpu,
            volume_gpu,
            laser_xyz_gpu,
            sensor_xyz_gpu,
            projector_focus_gpu,
            nt, nl, ns, nv,
            cp_t_start, cp_delta,
            int(t_accounts_first_and_last_bounces),
            int(camera_system.bp_accounts_for_d_2()),
            int(compensate_invsq),
            int(is_laser_paired_to_sensor),
            int(camera_system.is_transient()),
        ),
    )
    #cp.cuda.runtime.deviceSynchronize()

    H_1 = H_1_gpu.get()
    print(f'{volume_xyz.shape=}, {volume_xyz_shape=}, {H_1.shape=}')
    if camera_system.is_transient():
        H_1 = H_1.reshape((nt, *volume_xyz_shape))
    else:
        H_1 = H_1.reshape(volume_xyz_shape, order='A')

    return H_1