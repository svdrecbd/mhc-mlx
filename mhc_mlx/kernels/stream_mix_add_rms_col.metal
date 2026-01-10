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

// Load input column x[b, :, c] into registers
float x_col[MAX_N];
uint base_x = uint(b) * uint(n) * uint(C) + c;
uint stride_x = uint(C);

for (int j = 0; j < n; ++j) {
    x_col[j] = float(x[base_x + uint(j) * stride_x]);
}

// Precompute y_dist scalar part
// y_norm = y_agg[b,c] * inv_rms[b] * rms_weight[c]
uint idx_bc = uint(b) * uint(C) + c;
float y_norm_scalar = float(y_agg[idx_bc]) * float(inv_rms[b]) * float(rms_weight[c]);

// Compute Mix + Add
// out[i] = dot(M[i, :], x_col) + H_post[i] * y_norm_scalar
for (int i = 0; i < n; ++i) {
    float acc = 0.0f;
    int row_offset = i * n;
    
    // Matrix-vector dot product
    for (int j = 0; j < n; ++j) {
        acc += Ms[row_offset + j] * x_col[j];
    }
    
    // Add distributed branch
    float val = acc + Hpost[i] * y_norm_scalar;
    
    // Store result
    out[base_x + uint(i) * stride_x] = val;
}
