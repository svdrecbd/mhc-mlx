// MLX Metal kernel body (not a full .metal file).
//
// Fused backward for mHC:
// - computes y_agg, d_y_norm, inv_rms, d_r
// - computes dx
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32
// - M:          [n, n]    float32
// - H_pre:      [n]       float32 (activated)
// - H_post:     [n]       float32 (activated)
// - rms_weight: [C]       float32
// - d_out:      [B, n, C] float32
// - dx:         [B, n, C] float32
// - y_agg:      [B, C]    float32
// - d_y_norm:   [B, C]    float32
// - inv_rms:    [B]       float32
// - d_r:        [B]       float32

constexpr int MAX_N = {{MAX_N}};
constexpr int MAX_TPG = {{MAX_TPG}};
constexpr float EPS = {{EPS}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint token = thread_position_in_grid.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)token >= B) {
    return;
}

threadgroup float P[MAX_N * MAX_N];
threadgroup float Hpre[MAX_N];
threadgroup float Hpost[MAX_N];
threadgroup float reduce_buf[MAX_TPG];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += tpg) {
    P[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += tpg) {
    Hpre[idx] = H_pre[idx];
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = token * uint(n) * uint(C);
uint base_bc = token * uint(C);

float local_sum = 0.0f;
float local_dr = 0.0f;

for (uint c = lane; c < uint(C); c += tpg) {
    float y_agg_val = 0.0f;
    float d_y_norm_val = 0.0f;
    for (int i = 0; i < n; ++i) {
        uint idx = base + uint(i) * uint(C) + c;
        float xv = float(x[idx]);
        float doutv = float(d_out[idx]);
        y_agg_val += Hpre[i] * xv;
        d_y_norm_val += Hpost[i] * doutv;
    }
    y_agg[base_bc + c] = y_agg_val;
    d_y_norm[base_bc + c] = d_y_norm_val;
    local_sum += y_agg_val * y_agg_val;
    local_dr += d_y_norm_val * y_agg_val * float(rms_weight[c]);
}

reduce_buf[lane] = local_sum;
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

float mean_sq = reduce_buf[0] / float(C);
float inv = metal::rsqrt(mean_sq + EPS);

reduce_buf[lane] = local_dr;
threadgroup_barrier(mem_flags::mem_threadgroup);

active = tpg;
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

float dr = reduce_buf[0];
if (lane == 0u) {
    inv_rms[token] = inv;
    d_r[token] = dr;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float inv3 = inv * inv * inv;
float d_mean_sq = -0.5f * dr * inv3;
float scale = 2.0f / float(C);

for (uint c = lane; c < uint(C); c += tpg) {
    float y_agg_val = y_agg[base_bc + c];
    float d_y_norm_val = d_y_norm[base_bc + c];
    float d_y_agg = d_y_norm_val * inv * float(rms_weight[c])
        + d_mean_sq * scale * y_agg_val;

    for (int i = 0; i < n; ++i) {
        float dx_mix = 0.0f;
        for (int k_idx = 0; k_idx < n; ++k_idx) {
            dx_mix += P[k_idx * n + i] * float(d_out[base + uint(k_idx) * uint(C) + c]);
        }
        dx[base + uint(i) * uint(C) + c] = dx_mix + d_y_agg * Hpre[i];
    }
}
