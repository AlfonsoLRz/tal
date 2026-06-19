#include <cupy/complex.cuh>

typedef complex<float> complex_t;

#define EPS      0.00001f
#define FLT_MAX  3.402823466e+38F

__device__
complex_t complexMulScalar(complex_t a, float s) {
    return complex_t(a.real() * s, a.imag() * s);
}

__device__
complex_t complexAdd(complex_t a, complex_t b)
{
    return complex_t(a.real() + b.real(), a.imag() + b.imag());
}

__device__
complex_t complexLerp(complex_t a, complex_t b, float t) {
    return complexAdd(complexMulScalar(a, 1.0f - t), complexMulScalar(b, t));
}

__device__
float safeRCP(float x) {
    if (x > EPS || x < -EPS)
        return 1.0f / x;
    return x >= 0 ? FLT_MAX : -FLT_MAX;
}

__device__ __forceinline__
int idx3d(int x, int y, int z, int X, int Y, int Z) {
    return x + X * (y + Y * z);
}

__device__ __forceinline__
int wrapShiftedIndex(int index, int shift, int resolution)
{
    index += shift;
    return index >= resolution ? index - resolution : index;
}

extern "C" __global__
void stoltKernel(
        const complex_t* __restrict__ H,
        complex_t* __restrict__ result,
        int X, int Y, int Z,
        float invX, float invY, float invZ,
        int shiftX, int shiftY, int shiftZ,
        float stoltConstX,
        float stoltConstY
    )
{
    int x = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.x * blockDim.x + threadIdx.x + shiftZ;

    if (x >= X || y >= Y || z >= Z) return;

    // Normalized frequencies in [-1, 1]
    float fx = 2.0f * (x * invX) - 1.0f;
    float fy = 2.0f * (y * invY) - 1.0f;
    float fz = 2.0f * (z * invZ) - 1.0f;

    // Stolt mapping
    float sqrt_term = sqrtf(stoltConstX * fx * fx + stoltConstY * fy * fy + fz * fz);

    // fx/fy map back to exact integer bins after the cyclic shift, so only
    // the Stolt z coordinate needs interpolation.
    int x0 = wrapShiftedIndex(x, shiftX, X);
    int y0 = wrapShiftedIndex(y, shiftY, Y);
    float iz = (sqrt_term + 1.0f) * 0.5f * Z;

    iz = fmodf(iz + (float)shiftZ, (float)Z);

    int z0 = max(0, min((int)floorf(iz), Z - 1));
    int z1 = min(z0 + 1, Z - 1);
    float dz = iz - z0;

    complex_t c0 = H[idx3d(x0, y0, z0, X, Y, Z)];
    complex_t c1 = H[idx3d(x0, y0, z1, X, Y, Z)];
    complex_t res = complexLerp(c0, c1, dz);

    int outZ = wrapShiftedIndex(z, shiftZ, Z);
    result[idx3d(x0, y0, outZ, X, Y, Z)] = complexMulScalar(res, fabsf(fz) * safeRCP(sqrt_term));
}
