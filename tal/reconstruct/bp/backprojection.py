from tal.config import get_resources
from tal.log import log, LogLevel, TQDMLogRedirect
import cupy as cp
from numba import cuda
import numpy as np
import math
from tqdm import tqdm


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

    print(volume_xyz.shape)
    print(nt, nl, ns, nv)

    if is_laser_paired_to_sensor:
        assert laser_grid_xyz.shape[0] == ns, 'H does not match with laser_grid_xyz'
    else:
        assert laser_grid_xyz.shape[0] == nl, 'H does not match with laser_grid_xyz'

    assert sensor_grid_xyz.shape[0] == ns, 'H does not match with sensor_grid_xyz'
    assert (not t_accounts_first_and_last_bounces or (laser_xyz is not None and sensor_xyz is not None)), \
        't_accounts_first_and_last_bounces requires laser_xyz and sensor_xyz'

    print(laser_xyz.shape, sensor_xyz.shape)
    print(laser_grid_xyz.shape, sensor_grid_xyz.shape)
    
    # reshape everything into (nl, nv, ns, 3)
    if laser_xyz is not None:
        laser_xyz = laser_xyz.reshape((1, 1, 1, 3)).astype(np.float32)

    if sensor_xyz is not None:
        sensor_xyz = sensor_xyz.reshape((1, 1, 1, 3)).astype(np.float32)

    if is_laser_paired_to_sensor:
        laser_grid_xyz = laser_grid_xyz.reshape((1, 1, ns, 3)).astype(np.float32)
    else:
        laser_grid_xyz = laser_grid_xyz.reshape((nl, 1, 1, 3)).astype(np.float32)

    sensor_grid_xyz = sensor_grid_xyz.reshape((1, 1, ns, 3)).astype(np.float32)
    volume_xyz = volume_xyz.reshape((1, nv, 1, 3)).astype(np.float32)

    if camera_system.implements_projector():
        assert projector_focus is not None, 'projector_focus is required for this camera system'
        assert projector_focus.size == 3, \
            'When using tal.reconstruct.bp, projector_focus must be a single 3D point. ' \
            'If you want to focus the illumination aperture at multiple points, ' \
            'please use tal.reconstruct.pf_dev instead or call tal.reconstruct.bp once per projector_focus.'
        projector_focus = np.array(projector_focus).reshape((1, 1, 1, 3)).repeat(nv, axis=1)
    else:
        assert projector_focus is None, \
            'projector_focus must not be set for this camera system'
        projector_focus = volume_xyz.reshape((1, nv, 1, 3))

    projector_focus = cp.asarray(projector_focus)
    laser_grid_xyz = cp.asarray(laser_grid_xyz)
    sensor_grid_xyz = cp.asarray(sensor_grid_xyz)
    sensor_xyz = cp.asarray(sensor_xyz) if sensor_xyz is not None else None
    volume_xyz = cp.asarray(volume_xyz)

    def distance(a, b):
        return cp.linalg.norm(b - a, axis=-1)

    if camera_system.is_transient():
        H_1 = cp.zeros((nt, nv), dtype=H_0.dtype)
    else:
        H_1 = cp.zeros((nv, ), dtype=H_0.dtype)

    print(t_accounts_first_and_last_bounces)
    print(laser_xyz.shape, laser_grid_xyz.shape, sensor_grid_xyz.shape, sensor_xyz.shape, projector_focus.shape)

    # d_1: laser origin to laser illuminated point
    # d_2: laser illuminated point to projector_focus
    # d_3: x_v (camera_focus) to sensor imaged point
    # d_4: sensor imaged point to sensor origin
    if t_accounts_first_and_last_bounces:
        d_1 = distance(laser_xyz, laser_grid_xyz)
        d_4 = distance(sensor_grid_xyz, sensor_xyz)
    else:
        d_1 = cp.zeros((nl, 1, 1))
        d_4 = cp.zeros((1, 1, ns))

    print(d_1.shape, d_4.shape)

    if camera_system.bp_accounts_for_d_2():
        d_2 = distance(laser_grid_xyz, projector_focus)
    else:
        d_2 = cp.float32(0.0)

    H_0 = cp.asarray(H_0)
    d_1 = cp.asarray(d_1)
    d_2 = cp.asarray(d_2)
    d_4 = cp.asarray(d_4)

    threads_per_block = (16, 16, 4)
    blocks_per_grid = (
        (nv + threads_per_block[0] - 1) // threads_per_block[0],
        (nl + threads_per_block[1] - 1) // threads_per_block[1],
        (ns + threads_per_block[2] - 1) // threads_per_block[2],
    )
    
    backproject_numba[blocks_per_grid, threads_per_block](H_0, d_1, d_2, d_4, sensor_grid_xyz, volume_xyz, H_1,
                                                            t_start, delta_t, compensate_invsq, is_laser_paired_to_sensor,
                                                            camera_system.is_transient(),)

    H_1 = H_1.get()

    if camera_system.is_transient():
        H_1 = H_1.reshape((nt, *volume_xyz_shape))
    else:
        H_1 = H_1.reshape(volume_xyz_shape)

    return H_1


@cuda.jit(device=True)
def calculate_distance_squared(a, b):
    """Squared distance between two 3D points."""
    return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2


@cuda.jit
def backproject_numba(
        d_H_0,          # [nt, nl, ns]
        d_d_1,          # [nl, 1, 1]
        d_d_2,          # [nl, nv] or [nl, nv, ns]
        d_d_4,          # [1, 1, ns]
        d_sensor_pos,   # [1, 1, ns, 3]
        d_volume_xyz,   # [1, nv, 1, 3] depends on depths values
        d_H_1,          # [nt, nv]
        t_start,        # float
        delta_t,        # float
        compensate_invsq=False,
        is_laser_paired_to_sensor=False,
        is_transient=False
):
    """
    Backprojection kernel for NLOS reconstruction.
    This kernel computes the backprojection of the input data H_0
    to the volume defined by volume_xyz."""
    nt, nl, ns = d_H_0.shape
    nv = d_volume_xyz.shape[1]
    epsilon = 1e-4
    
    v = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    l = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    s = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    
    if v >= nv or l >= nl or s >= ns:
        return

    # d_1: laser origin to laser illuminated point
    # d_2: laser illuminated point to projector_focus
    # d_3: x_v (camera_focus) to sensor imaged point
    # d_4: sensor imaged point to sensor origin
    d1 = d_d_1[l, 0, 0]
    d4 = d_d_4[0, 0, s]
    d2 = d_d_2[l, v, s] if is_laser_paired_to_sensor else d_d_2[l, v, 0]
    
    # Optimized distance calculation
    volume_pos = (d_volume_xyz[0, v, 0, 0], 
                 d_volume_xyz[0, v, 0, 1], 
                 d_volume_xyz[0, v, 0, 2])
    sensor_pos = (d_sensor_pos[0, 0, s, 0],
                 d_sensor_pos[0, 0, s, 1],
                 d_sensor_pos[0, 0, s, 2])
    
    d3_sq = calculate_distance_squared(volume_pos, sensor_pos)
    d3 = math.sqrt(d3_sq)

    # Time index
    total_dist = d1 + d2 + d3 + d4
    idx = int((total_dist - t_start) / delta_t)
    
    # Compensation
    invsq = 1.0
    if compensate_invsq:
        denominator = max(d1 * d2 * d3 * d4, epsilon)
        invsq = 1.0 / denominator

    t_range = nt if is_transient else 1
    for t in range(t_range):
        idx_t = idx + t
        if 0 <= idx_t < nt:
            val = d_H_0[idx_t, l, s] * invsq if compensate_invsq else d_H_0[idx_t, l, s]
            cuda.atomic.add(d_H_1, v, val)