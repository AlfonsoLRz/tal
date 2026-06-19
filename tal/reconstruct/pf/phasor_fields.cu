#include <cupy/complex.cuh>

typedef complex<float> complex_t;

extern "C" __global__
void rsdKernel(
        const float* __restrict__ coords,
        const float* __restrict__ omega,
        complex_t* __restrict__ out,
        int nw, int nx, int ny,
        float depth,
        int twice
    )
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;

    if (w >= nw || x >= nx || y >= ny) return;

    int cidx = (x * ny + y) * 3;
    float dx = coords[cidx + 0];
    float dy = coords[cidx + 1];
    float dz = coords[cidx + 2] + depth;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    float inv_dist = 1.0f / dist;
    float phase = omega[w] * dist;

    float s, c;
    sincosf(twice ? 2.0f * phase : phase, &s, &c);
    float amp = twice ? inv_dist * inv_dist : inv_dist;
    out[y + ny * (x + nx * w)] = complex_t(c * amp, s * amp);
}
