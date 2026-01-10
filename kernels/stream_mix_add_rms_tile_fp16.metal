// MLX Metal kernel body (not a full .metal file).
//
// Half2-tiled stream mix + RMS add for fixed n (TILE_N).
// Computes a [TILE_N x (TILE_C*2)] block of output per threadgroup.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float16
// - M:          [n, n]    float32
// - H_post:     [n]       float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] float16

constexpr int TILE_N = {{TILE_N}};
constexpr int TILE_C = {{TILE_C}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint gid_x = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)b >= B) {
    return;
}

uint tile = gid_x / tpg;
uint c2_offset = lane % TILE_C;
uint i = lane / TILE_C;

if ((int)i >= n) {
    return;
}

uint c2 = tile * (TILE_C * 2u) + c2_offset * 2u;

threadgroup float M_tile[TILE_N * TILE_N];
threadgroup float Hpost[TILE_N];
threadgroup float2 X_tile[TILE_N * TILE_C];

uint mn = uint(n * n);
for (uint idx = lane; idx < mn; idx += tpg) {
    M_tile[idx] = M[idx];
}
if (lane < uint(n)) {
    Hpost[lane] = H_post[lane];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

bool has0 = (int)c2 < C;
bool has1 = (int)(c2 + 1u) < C;

float2 xval = float2(0.0f);
if (has0) {
    uint idx = base + uint(i) * uint(C) + c2;
    if (has1) {
        if ((idx & 1u) == 0u) {
            const device packed_half2* x_ptr = (const device packed_half2*)(x + idx);
            half2 xv = half2(*x_ptr);
            xval = float2(float(xv.x), float(xv.y));
        } else {
            xval = float2(float(x[idx]), float(x[idx + 1u]));
        }
    } else {
        xval = float2(float(x[idx]), 0.0f);
    }
}
X_tile[uint(i) * TILE_C + c2_offset] = xval;
threadgroup_barrier(mem_flags::mem_threadgroup);

float2 acc = float2(0.0f);
for (int j = 0; j < n; ++j) {
    acc += M_tile[i * n + j] * X_tile[uint(j) * TILE_C + c2_offset];
}

if (has0) {
    float inv = float(inv_rms[b]);
    float y0 = float(y_agg[base_bc + c2]) * inv * float(rms_weight[c2]);
    float out0 = acc.x + Hpost[i] * y0;
    uint out_idx = base + uint(i) * uint(C) + c2;
    if (has1) {
        float y1 = float(y_agg[base_bc + c2 + 1u]) * inv * float(rms_weight[c2 + 1u]);
        float out1 = acc.y + Hpost[i] * y1;
        device packed_half2* out_ptr = (device packed_half2*)(out + out_idx);
        *out_ptr = packed_half2(half2(half(out0), half(out1)));
    } else {
        out[out_idx] = half(out0);
    }
}
