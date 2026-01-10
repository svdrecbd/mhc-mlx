// MLX Metal kernel body (not a full .metal file).
//
// Reduces partial sums to inv_rms and d_r:
// inv_rms[b] = rsqrt(mean(y_agg[b, :]^2) + eps)
// d_r[b] = sum_c d_y_norm[b, c] * y_agg[b, c] * rms_weight[c]
//
// Expected row-contiguous shapes:
// - y_agg:      [B, C] float32
// - partial_sq: [B, T] float32
// - partial_dr: [B, T] float32
// - inv_rms:    [B]    float32
// - d_r:        [B]    float32

constexpr int MAX_TPG = {{MAX_TPG}};
constexpr float EPS = {{EPS}};

uint lane = thread_position_in_threadgroup.x;
uint b = thread_position_in_grid.y;
uint tpg = threads_per_threadgroup.x;

int B = y_agg_shape[0];
int C = y_agg_shape[1];
int tiles = partial_sq_shape[1];

if ((int)b >= B) {
    return;
}

threadgroup float reduce_sq[MAX_TPG];
threadgroup float reduce_dr[MAX_TPG];

float partial_sq_sum = 0.0f;
float partial_dr_sum = 0.0f;
for (uint t = lane; t < (uint)tiles; t += tpg) {
    uint idx = uint(b) * uint(tiles) + t;
    partial_sq_sum += partial_sq[idx];
    partial_dr_sum += partial_dr[idx];
}

reduce_sq[lane] = partial_sq_sum;
reduce_dr[lane] = partial_dr_sum;
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
    float mean_sq = reduce_sq[0] / float(C);
    inv_rms[b] = metal::rsqrt(mean_sq + EPS);
    d_r[b] = reduce_dr[0];
}
