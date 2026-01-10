// MLX Metal kernel body (not a full .metal file).
//
// Optimized "Column-Parallel" Backward DX.
//
// Computes dx for the fused mHC forward:
// dx = M^T * d_out + d_y_agg * H_pre
//
// Optimization:
// - Loads d_out[b, :, c] (entire column) into registers once.
// - Performs M^T @ d_out_col in registers.
// - Reduces d_out reads from O(n^2) to O(n) per column.
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
uint b = thread_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if (b >= (uint)B || c >= (uint)C) {
    return;
}

// Load M and H_pre into threadgroup memory
threadgroup float Ms[MAX_N * MAX_N];
threadgroup float Hpre[MAX_N];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += threads_per_threadgroup.x) {
    Ms[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += threads_per_threadgroup.x) {
    Hpre[idx] = H_pre[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Compute d_y_agg scalar part
// d_y_agg = d_y_norm * inv * rms_weight + d_mean_sq * (2/C) * y_agg
float inv = float(inv_rms[b]);
float dr = float(d_r[b]);
float inv3 = inv * inv * inv;
float d_mean_sq = -0.5f * dr * inv3;

uint idx_bc = uint(b) * uint(C) + c;
float y_agg_val = float(y_agg[idx_bc]);
float d_y_norm_val = float(d_y_norm[idx_bc]);
float d_y_agg = d_y_norm_val * inv * float(rms_weight[c])
    + d_mean_sq * (2.0f / float(C)) * y_agg_val;

// Load d_out column d_out[b, :, c] into registers
float d_out_col[MAX_N];
uint base = uint(b) * uint(n) * uint(C) + c;
uint stride = uint(C);

for (int k = 0; k < n; ++k) {
    d_out_col[k] = float(d_out[base + uint(k) * stride]);
}

// Compute dx
// dx[i] = (sum_k M[k, i] * d_out[k]) + d_y_agg * H_pre[i]
// Note: We need M^T, so we sum over rows of M column i.
// M is [n, n], flat index [k*n + i]

for (int i = 0; i < n; ++i) {
    float dx_mix = 0.0f;
    for (int k = 0; k < n; ++k) {
        // M[k, i] is Ms[k * n + i]
        dx_mix += Ms[k * n + i] * d_out_col[k];
    }
    
    float val = dx_mix + d_y_agg * Hpre[i];
    dx[base + uint(i) * stride] = val;
}
