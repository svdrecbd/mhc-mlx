// MLX Metal kernel body (not a full .metal file).
//
// Computes dx for the fused mHC forward:
// dx = M^T * d_out + d_y_agg * H_pre
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32
// - M:          [n, n]    float32
// - H_pre:      [n]       float32 (activated)
// - rms_weight: [C]       float32
// - d_out:      [B, n, C] float32
// - y_agg:      [B, C]    float32
// - d_y_norm:   [B, C]    float32
// - inv_rms:    [B]       float32
// - d_r:        [B]       float32
// - dx:         [B, n, C] float32

constexpr int MAX_N = {{MAX_N}};

uint c = thread_position_in_grid.x;
uint k = thread_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

int b = (int)(k / (uint)n);
int i = (int)(k - (uint)(b * n));

if (b >= B || (int)c >= C) {
    return;
}

threadgroup float P[MAX_N * MAX_N];
threadgroup float Hpre[MAX_N];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += threads_per_threadgroup.x) {
    P[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += threads_per_threadgroup.x) {
    Hpre[idx] = H_pre[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

float inv = inv_rms[b];
float dr = d_r[b];
float inv3 = inv * inv * inv;
float d_mean_sq = -0.5f * dr * inv3;

float y_agg_val = y_agg[base_bc + c];
float d_y_norm_val = d_y_norm[base_bc + c];
float d_y_agg = d_y_norm_val * inv * float(rms_weight[c])
    + d_mean_sq * (2.0f / float(C)) * y_agg_val;

float dx_agg = d_y_agg * Hpre[i];

float dx_mix = 0.0f;
for (int k_idx = 0; k_idx < n; ++k_idx) {
    dx_mix += P[k_idx * n + i] * float(d_out[base + uint(k_idx) * uint(C) + c]);
}

dx[base + uint(i) * uint(C) + c] = dx_mix + dx_agg;
