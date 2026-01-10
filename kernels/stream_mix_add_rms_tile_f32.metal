// MLX Metal kernel body (not a full .metal file).
//
// Tiled stream mix + RMS add for fixed n (TILE_N).
// Computes a [TILE_N x TILE_C] block of output per threadgroup.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32/float16
// - M:          [n, n]    float32
// - H_post:     [n]       float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] float32

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
uint c_offset = lane % TILE_C;
uint i = lane / TILE_C;

if ((int)i >= n) {
    return;
}

uint c = tile * TILE_C + c_offset;

threadgroup float M_tile[TILE_N * TILE_N];
threadgroup float Hpost[TILE_N];
threadgroup float X_tile[TILE_N * TILE_C];

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

float xval = 0.0f;
if ((int)c < C) {
    xval = float(x[base + uint(i) * uint(C) + c]);
}
X_tile[uint(i) * TILE_C + c_offset] = xval;
threadgroup_barrier(mem_flags::mem_threadgroup);

float acc = 0.0f;
for (int j = 0; j < n; ++j) {
    acc += M_tile[i * n + j] * X_tile[uint(j) * TILE_C + c_offset];
}

if ((int)c < C) {
    float y = float(y_agg[base_bc + c]) * float(inv_rms[b]) * float(rms_weight[c]);
    uint out_idx = base + uint(i) * uint(C) + c;
    out[out_idx] = acc + Hpost[i] * y;
}
