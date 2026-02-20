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
