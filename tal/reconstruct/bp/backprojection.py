"""
Docstring for tal.reconstruct.bp.backprojection
"""

from tal.config import get_resources
from tal.log import log, LogLevel, TQDMLogRedirect
import cupy as cp
from numba import cuda
import numpy as np


# CUDA kernel as in the C++ code; hope this will be easy to maintain both C++ and Python versions in the future

bp_kernel_source = r"""
extern "C" __global__
void backproject_kernel(
    const float* __restrict__ H0,               // [nt, nl, ns]
    float* __restrict__ H1,                     // [nt, nv] or [nv] depending on transient
    const float* __restrict__ laser_grid,       // [nl, 3] or [ns, 3] depending
    const float* __restrict__ sensor_grid,      // [ns, 3]
    const float* __restrict__ volume_xyz,       // [nv, 3]
    const float* __restrict__ laser_xyz,        // [3]
    const float* __restrict__ sensor_xyz,       // [3]
    const float* __restrict__ projector_focus,  // [nv, 3]
    const int nt,
    const int nl,
    const int ns,
    const int nv,
    const float t_start,
    const float delta_t,
    const int t_accounts_first_and_last_bounces,
    const int bp_accounts_for_d_2,
    const int compensate_invsq,
    const int is_laser_paired_to_sensor,
    const int is_transient
){
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.z * blockDim.z + threadIdx.z;

    if (v >= nv || l >= nl || s >= ns)
        return;

    float vx = volume_xyz[3*v + 0];
    float vy = volume_xyz[3*v + 1];
    float vz = volume_xyz[3*v + 2];

    float sx = sensor_grid[3*s + 0];
    float sy = sensor_grid[3*s + 1];
    float sz = sensor_grid[3*s + 2];

    int li = (is_laser_paired_to_sensor ? s : l);
    float lx = laser_grid[3*li + 0];
    float ly = laser_grid[3*li + 1];
    float lz = laser_grid[3*li + 2];

    float px = projector_focus[3*v + 0];
    float py = projector_focus[3*v + 1];
    float pz = projector_focus[3*v + 2];

    float d1 = 0.0f;
    if (t_accounts_first_and_last_bounces && laser_xyz != nullptr)
    {
        float dx1 = lx - laser_xyz[0];
        float dy1 = ly - laser_xyz[1];
        float dz1 = lz - laser_xyz[2];
        d1 = sqrtf(dx1*dx1 + dy1*dy1 + dz1*dz1);
    }

    float d2 = 0.0f;
    if (bp_accounts_for_d_2) 
    {
        float dx2 = px - lx;
        float dy2 = py - ly;
        float dz2 = pz - lz;
        d2 = sqrtf(dx2*dx2 + dy2*dy2 + dz2*dz2);
    }

    float dx3 = sx - vx;
    float dy3 = sy - vy;
    float dz3 = sz - vz;
    float d3 = sqrtf(dx3*dx3 + dy3*dy3 + dz3*dz3);

    float d4 = 0.0f;
    if (t_accounts_first_and_last_bounces && sensor_xyz != nullptr)
    {
        float dx4 = sensor_xyz[0] - sx;
        float dy4 = sensor_xyz[1] - sy;
        float dz4 = sensor_xyz[2] - sz;
        d4 = sqrtf(dx4*dx4 + dy4*dy4 + dz4*dz4);
    }

    float total_dist = d1 + d2 + d3 + d4;

    int idx = (int)((total_dist - t_start) / delta_t);

    float invsq = 1.0f;
    if (compensate_invsq) 
    {
        float denom = d1 * d2 * d3 * d4 + 1e-5f;
        invsq = 1.0f / denom;
    }

    if (is_transient) 
    {
        for (int t = 0; t < nt; ++t) 
        {
            int idx_t = idx + t;
            if (0 <= idx_t && idx_t < nt) {
                float val = H0[(idx_t * nl + l) * ns + s] * invsq;
                atomicAdd(&H1[t * nv + v], val);
            }
        }
    } 
    else 
    {
        if (0 <= idx && idx < nt) 
        {
            float val = H0[(idx * nl + l) * ns + s] * invsq;
            atomicAdd(&H1[v], val);
        }
    }
}
"""

# Compile only once                             
bp_kernel = cp.RawKernel(
    bp_kernel_source,
    "backproject_kernel",
    options=(
        "--use_fast_math",
#        "-arch=compute_89",
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
    if camera_system.is_transient():
        H_1 = H_1.reshape((nt, *volume_xyz_shape))
    else:
        H_1 = H_1.reshape(volume_xyz_shape)

    return H_1