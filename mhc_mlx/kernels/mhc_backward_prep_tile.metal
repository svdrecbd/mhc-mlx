// MLX Metal kernel body (not a full .metal file).
//
// Tile-parallel backward prep:
// - y_agg[b, c] = sum_i H_pre[i] * x[b, i, c]
// - d_y_norm[b, c] = sum_i H_post[i] * d_out[b, i, c]
// - partial_sq[b, tile] = sum_{c in tile} y_agg[b, c]^2
// - partial_dr[b, tile] = sum_{c in tile} d_y_norm[b, c] * y_agg[b, c] * rms_weight[c]
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32/float16
// - H_pre:      [n]       float32 (activated)
// - H_post:     [n]       float32 (activated)
// - rms_weight: [C]       float32
// - d_out:      [B, n, C] float32/float16
// - y_agg:      [B, C]    float32
// - d_y_norm:   [B, C]    float32
// - partial_sq: [B, T]    float32
// - partial_dr: [B, T]    float32

constexpr int MAX_N = {{MAX_N}};
constexpr int MAX_TPG = {{MAX_TPG}};

uint lane = thread_position_in_threadgroup.x;
uint c = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;
uint tpg = threads_per_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)b >= B) {
    return;
}

uint tiles = (uint(C) + tpg - 1u) / tpg;
uint tile = c / tpg;

threadgroup float Hpre[MAX_N];
threadgroup float Hpost[MAX_N];
threadgroup float reduce_sq[MAX_TPG];
threadgroup float reduce_dr[MAX_TPG];

for (uint idx = lane; idx < uint(n); idx += tpg) {
    Hpre[idx] = H_pre[idx];
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

int c_i = int(c);
uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

float y_val = 0.0f;
float d_y_norm_val = 0.0f;
if (c_i < C) {
    for (int i = 0; i < n; ++i) {
        uint idx = base + uint(i) * uint(C) + uint(c_i);
        y_val += Hpre[i] * float(x[idx]);
        d_y_norm_val += Hpost[i] * float(d_out[idx]);
    }
    y_agg[base_bc + uint(c_i)] = y_val;
    d_y_norm[base_bc + uint(c_i)] = d_y_norm_val;
}

float local_sq = (c_i < C) ? (y_val * y_val) : 0.0f;
float local_dr = 0.0f;
if (c_i < C) {
    local_dr = d_y_norm_val * y_val * float(rms_weight[c_i]);
}

reduce_sq[lane] = local_sq;
reduce_dr[lane] = local_dr;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_sq[lane] += reduce_sq[lane + stride];
        reduce_dr[lane] += reduce_dr[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_sq[0] += reduce_sq[active - 1u];
        reduce_dr[0] += reduce_dr[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

if (lane == 0u) {
    uint out_idx = uint(b) * tiles + tile;
    partial_sq[out_idx] = reduce_sq[0];
    partial_dr[out_idx] = reduce_dr[0];
}
