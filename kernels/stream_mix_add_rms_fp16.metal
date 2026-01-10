// MLX Metal kernel body (not a full .metal file).
//
// Half2-optimized stream mix + RMS add, writing float16 output.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float16
// - M:          [n, n]    float32
// - H_post:     [n]       float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] float16

constexpr int MAX_N = {{MAX_N}};

uint c2 = thread_position_in_grid.x * 2u;
uint k = thread_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

int b = (int)(k / (uint)n);
int i = (int)(k - (uint)(b * n));

if (b >= B || (int)c2 >= C) {
    return;
}

bool has_second = (int)(c2 + 1u) < C;

threadgroup float Ms[MAX_N * MAX_N];
threadgroup float Hpost[MAX_N];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += threads_per_threadgroup.x) {
    Ms[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += threads_per_threadgroup.x) {
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

uint out_idx = base + uint(i) * uint(C) + c2;
float inv = float(inv_rms[b]);

if (has_second) {
    float2 acc = float2(0.0f);
    for (int j = 0; j < n; ++j) {
        uint idx = base + uint(j) * uint(C) + c2;
        const device packed_half2* x_ptr = (const device packed_half2*)(x + idx);
        half2 xv = half2(*x_ptr);
        float2 xf = float2(float(xv.x), float(xv.y));
        acc += Ms[i * n + j] * xf;
    }

    float y0 = float(y_agg[base_bc + c2]) * inv * float(rms_weight[c2]);
    float y1 = float(y_agg[base_bc + c2 + 1u]) * inv * float(rms_weight[c2 + 1u]);
    float out0 = acc.x + Hpost[i] * y0;
    float out1 = acc.y + Hpost[i] * y1;

    device packed_half2* out_ptr = (device packed_half2*)(out + out_idx);
    *out_ptr = packed_half2(half2(half(out0), half(out1)));
} else {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
        uint idx = base + uint(j) * uint(C) + c2;
        acc += Ms[i * n + j] * float(x[idx]);
    }
    float y0 = float(y_agg[base_bc + c2]) * inv * float(rms_weight[c2]);
    out[out_idx] = half(acc + Hpost[i] * y0);
}
