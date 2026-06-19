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
void shiftedIdx3D(int& x, int& y, int& z,
                  int X, int Y, int Z,
                  int shiftX, int shiftY, int shiftZ)
{
    x = (x + shiftX) % X;
    y = (y + shiftY) % Y;
    z = (z + shiftZ) % Z;
}

extern "C" __global__
void stoltKernel(
        const complex_t* __restrict__ H,
        complex_t* __restrict__ result,
        int X, int Y, int Z,
        float invX, float invY, float invZ,
        int shiftX, int shiftY, int shiftZ,
        float stoltConst
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
    float sqrt_term = sqrtf(stoltConst * (fx * fx + fy * fy) + fz * fz);

    // Map to index space
    float ix = (fx + 1.0f) * 0.5f * X;
    float iy = (fy + 1.0f) * 0.5f * Y;
    float iz = (sqrt_term + 1.0f) * 0.5f * Z;

    // Cyclic shift in frequency space (equivalent to fftshift/ifftshift)
    ix = fmodf(ix + (float)shiftX, (float)X);
    iy = fmodf(iy + (float)shiftY, (float)Y);
    iz = fmodf(iz + (float)shiftZ, (float)Z);

    int x0 = max(0, min((int)floorf(ix), X - 1));
    int y0 = max(0, min((int)floorf(iy), Y - 1));
    int z0 = max(0, min((int)floorf(iz), Z - 1));

    int x1 = min(x0 + 1, X - 1);
    int y1 = min(y0 + 1, Y - 1);
    int z1 = min(z0 + 1, Z - 1);

    float dx = ix - x0;
    float dy = iy - y0;
    float dz = iz - z0;

    complex_t c000 = H[idx3d(x0, y0, z0, X, Y, Z)];
    complex_t c100 = H[idx3d(x1, y0, z0, X, Y, Z)];
    complex_t c010 = H[idx3d(x0, y1, z0, X, Y, Z)];
    complex_t c110 = H[idx3d(x1, y1, z0, X, Y, Z)];
    complex_t c001 = H[idx3d(x0, y0, z1, X, Y, Z)];
    complex_t c101 = H[idx3d(x1, y0, z1, X, Y, Z)];
    complex_t c011 = H[idx3d(x0, y1, z1, X, Y, Z)];
    complex_t c111 = H[idx3d(x1, y1, z1, X, Y, Z)];

    complex_t c00 = complexLerp(c000, c100, dx);
    complex_t c10 = complexLerp(c010, c110, dx);
    complex_t c01 = complexLerp(c001, c101, dx);
    complex_t c11 = complexLerp(c011, c111, dx);

    complex_t c0 = complexLerp(c00, c10, dy);
    complex_t c1 = complexLerp(c01, c11, dy);
    complex_t res = complexLerp(c0, c1, dz);

    // Cyclic shift in spatial index space
    shiftedIdx3D(x, y, z, X, Y, Z, shiftX, shiftY, shiftZ);
    result[idx3d(x, y, z, X, Y, Z)] = complexMulScalar(res, fabsf(fz) * safeRCP(sqrt_term));
}
