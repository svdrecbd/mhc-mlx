// MLX Metal kernel body (not a full .metal file).
//
// Tile-parallel variant of stream_mix_add_rms using a 2D threadgroup.
// Threadgroup.y == n so a group computes all i in [0,n) for a batch b
// and a tile of channels, amortizing M and H_post loads.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32/float16
// - M:          [n, n]    float32
// - H_post:     [n]       float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] float32

constexpr int MAX_N = {{MAX_N}};

uint c = thread_position_in_grid.x;
uint k = thread_position_in_grid.y;
uint lane_x = thread_position_in_threadgroup.x;
uint lane_y = thread_position_in_threadgroup.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

int b = (int)(k / (uint)n);
int i = (int)(k - (uint)(b * n));

if (b >= B || i >= n || (int)c >= C) {
    return;
}

threadgroup float Ms[MAX_N * MAX_N];
threadgroup float Hpost[MAX_N];

uint tpg_x = threads_per_threadgroup.x;
uint tpg_y = threads_per_threadgroup.y;
uint tcount = tpg_x * tpg_y;
uint lane = lane_y * tpg_x + lane_x;

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += tcount) {
    Ms[idx] = M[idx];
}
for (uint idx = lane; idx < (uint)n; idx += tcount) {
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

float acc = 0.0f;
for (int j = 0; j < n; ++j) {
    acc += Ms[i * n + j] * float(x[base + uint(j) * uint(C) + c]);
}

float y = float(y_agg[base_bc + c]) * float(inv_rms[b]) * float(rms_weight[c]);
uint out_idx = base + uint(i) * uint(C) + c;
out[out_idx] = acc + Hpost[i] * y;
