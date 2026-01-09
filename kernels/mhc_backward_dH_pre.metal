// MLX Metal kernel body (not a full .metal file).
//
// Computes dH_pre for the aggregate branch:
// dH_pre[i] = sum_{b, c} d_y_agg[b, c] * x[b, i, c]
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32
// - y_agg:      [B, C]    float32
// - d_y_norm:   [B, C]    float32
// - inv_rms:    [B]       float32
// - d_r:        [B]       float32
// - rms_weight: [C]       float32
// - dH_pre:     [n]       float32

constexpr int MAX_TPG = {{MAX_TPG}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint i = thread_position_in_grid.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)i >= n) {
    return;
}

threadgroup float reduce_buf[MAX_TPG];

float partial = 0.0f;
for (int b = 0; b < B; ++b) {
    float inv = inv_rms[b];
    float dr = d_r[b];
    float inv3 = inv * inv * inv;
    float d_mean_sq = -0.5f * dr * inv3;
    uint base = uint(b) * uint(n) * uint(C);
    uint base_bc = uint(b) * uint(C);
    for (uint c = lane; c < uint(C); c += tpg) {
        float y_agg_val = y_agg[base_bc + c];
        float d_y_norm_val = d_y_norm[base_bc + c];
        float d_y_agg = d_y_norm_val * inv * float(rms_weight[c])
            + d_mean_sq * (2.0f / float(C)) * y_agg_val;
        float xv = float(x[base + uint(i) * uint(C) + c]);
        partial += d_y_agg * xv;
    }
}

reduce_buf[lane] = partial;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_buf[lane] += reduce_buf[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_buf[0] += reduce_buf[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

if (lane == 0u) {
    dH_pre[i] = reduce_buf[0];
}
