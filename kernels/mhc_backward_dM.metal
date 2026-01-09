// MLX Metal kernel body (not a full .metal file).
//
// Computes dM for stream mixing:
// dM[i, j] = sum_{b, c} d_out[b, i, c] * x[b, j, c]
//
// Expected row-contiguous shapes:
// - x:     [B, n, C] float32
// - d_out: [B, n, C] float32
// - dM:    [n, n]    float32

constexpr int MAX_TPG = {{MAX_TPG}};
constexpr int MAX_SIMDGROUPS = (MAX_TPG + 31) / 32;

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint idx = thread_position_in_grid.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

int nn = n * n;
if ((int)idx >= nn) {
    return;
}

int i = (int)(idx / (uint)n);
int j = (int)(idx - (uint)(i * n));

threadgroup float simd_buf[MAX_SIMDGROUPS];

float partial = 0.0f;
uint C4 = (uint(C) / 4u) * 4u;

for (int b = 0; b < B; ++b) {
    uint base = uint(b) * uint(n) * uint(C);
    uint base_i = base + uint(i) * uint(C);
    uint base_j = base + uint(j) * uint(C);

    for (uint c = lane * 4u; c < C4; c += tpg * 4u) {
        const device packed_float4* dout_ptr =
            (const device packed_float4*)(d_out + base_i + c);
        const device packed_float4* x_ptr =
            (const device packed_float4*)(x + base_j + c);
        float4 doutv = float4(*dout_ptr);
        float4 xv = float4(*x_ptr);
        partial += dot(doutv, xv);
    }

    for (uint c = C4 + lane; c < uint(C); c += tpg) {
        float doutv = float(d_out[base_i + c]);
        float xv = float(x[base_j + c]);
        partial += doutv * xv;
    }
}

float simd_sum = metal::simd_sum(partial);
if (thread_index_in_simdgroup == 0u) {
    simd_buf[simdgroup_index_in_threadgroup] = simd_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (lane == 0u) {
    float total = 0.0f;
    uint simd_groups = (tpg + 31u) / 32u;
    for (uint g = 0; g < simd_groups; ++g) {
        total += simd_buf[g];
    }
    dM[idx] = total;
}
