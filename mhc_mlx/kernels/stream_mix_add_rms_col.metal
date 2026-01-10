// MLX Metal kernel body (not a full .metal file).
//
// Optimized "Column-Parallel" Mix + Add + RMS Apply.
//
// Fuses:
// - stream mixing: out[b,i,c] = sum_j M[i,j] * x[b,j,c]
// - final add: out += H_post[i] * (y_agg[b,c] * inv_rms[b] * rms_weight[c])
//
// Optimization:
// - Loads x[b, :, c] (entire column) into registers once.
// - Performs M @ x_col in registers.
// - Reduces x reads from O(n^2) to O(n) per column (factor of n reduction).
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32/float16
// - M:          [n, n]    float32
// - H_post:     [n]       float32 (activated)
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] float32

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

// Load M and H_post into threadgroup memory for shared access
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

uint base_token = b * uint(n) * uint(C);
uint base_token_y = b * uint(C);
float inv_rms_val = inv_rms[b];

// Load x column into registers
float x_col[MAX_N];
for (int i = 0; i < n; ++i) {
    x_col[i] = float(x[base_token + uint(i) * uint(C) + c]);
}

float y_val = y_agg[base_token_y + c];
float y_n = y_val * inv_rms_val * float(rms_weight[c]);

// Compute M * x_col + H_post * y_norm
for (int i = 0; i < n; ++i) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    int row_offset = i * n;
    
    // Unroll 2x (8 elements per step) for ILP
    for (int j = 0; j < n; j += 8) {
        float4 m_vec0 = float4(Ms[row_offset + j], Ms[row_offset + j + 1], Ms[row_offset + j + 2], Ms[row_offset + j + 3]);
        float4 x_vec0 = float4(x_col[j], x_col[j + 1], x_col[j + 2], x_col[j + 3]);
        acc0 += dot(m_vec0, x_vec0);

        float4 m_vec1 = float4(Ms[row_offset + j + 4], Ms[row_offset + j + 5], Ms[row_offset + j + 6], Ms[row_offset + j + 7]);
        float4 x_vec1 = float4(x_col[j + 4], x_col[j + 5], x_col[j + 6], x_col[j + 7]);
        acc1 += dot(m_vec1, x_vec1);
    }
    
    out[base_token + uint(i) * uint(C) + c] = (acc0 + acc1) + Hpost[i] * y_n;
}

